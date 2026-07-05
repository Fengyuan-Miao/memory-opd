# Copyright 2025 Individual Contributor: Fengyuan Miao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Step-level OPD-MM teacher correction collection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from verl.experimental.opd_mm.executor import ToolExecutor
from verl.experimental.opd_mm.models import OPDSample, PolicyOutput, SFTExample, ToolAction
from verl.experimental.opd_mm.on_policy_distiller import AnswerJudge, AnswerModel
from verl.experimental.opd_mm.retrieval import TurnAwareHybridRetriever
from verl.experimental.opd_mm.schema import TrajectoryValidator
from verl.experimental.opd_mm.tools import OPDToolSession


class StepTeacherPolicy(Protocol):
    """Teacher interface for one-step correction at an on-policy state."""

    def correct_next(
        self,
        query: str,
        gold_answer: str,
        history: list[ToolAction],
        observation: dict[str, Any],
        feedback: dict[str, Any],
        privileged_context: Optional[dict[str, Any]] = None,
    ) -> PolicyOutput | list[ToolAction]:
        """Return the corrected next action chunk for the current state."""
        ...


@dataclass
class StepCorrection:
    """One state-level corrected target."""

    sample_id: str
    step_index: int
    history: list[ToolAction]
    observation: dict[str, Any]
    feedback: dict[str, Any]
    teacher_actions: list[ToolAction]
    example: SFTExample
    student_next_action: Optional[ToolAction] = None
    teacher_raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "step_index": self.step_index,
            "history": [action.to_dict() for action in self.history],
            "observation": self.observation,
            "feedback": self.feedback,
            "teacher_actions": [action.to_dict() for action in self.teacher_actions],
            "student_next_action": self.student_next_action.to_dict() if self.student_next_action else None,
            "teacher_raw_response": self.teacher_raw_response,
            "example": self.example.to_dict(include_metadata=True),
            "metadata": self.metadata,
        }


def build_step_student_prompt(
    query: str,
    history: list[ToolAction],
    observation: dict[str, Any],
    schema: str,
) -> str:
    """Build the student-visible prompt for one OPD-MM state."""
    return f"""You are a multimodal memory retrieval planner.
Choose the next executable tool call for the current retrieval state.

You cannot see the hidden memory store, memory index, candidate memory IDs, gold answer, or teacher-only context.
Use only the allowed schema and the public observation.

{schema}

User query:
{query}

Previous actions:
{json.dumps([action.to_dict() for action in history], ensure_ascii=False, indent=2)}

Current observation:
{json.dumps(observation, ensure_ascii=False, indent=2, default=str)}
"""


class StepCorrectionCollector:
    """Collect corrected next-action labels on student-visited OPD-MM states."""

    def __init__(
        self,
        teacher: StepTeacherPolicy,
        executor: Optional[ToolExecutor] = None,
        answer_model: Optional[AnswerModel] = None,
        judge: Optional[AnswerJudge] = None,
        validator: Optional[TrajectoryValidator] = None,
        max_steps: int = 16,
        include_feedback: bool = False,
    ):
        self.teacher = teacher
        self.executor = executor or ToolExecutor(retriever=TurnAwareHybridRetriever())
        self.answer_model = answer_model
        self.judge = judge
        self.validator = validator or TrajectoryValidator()
        self.max_steps = max(1, int(max_steps))
        self.include_feedback = include_feedback

    def collect(
        self,
        sample: OPDSample,
        student_actions: list[ToolAction],
        *,
        round_index: int = 0,
    ) -> list[StepCorrection]:
        """Replay a student trajectory and collect teacher-corrected next actions."""
        session = OPDToolSession(
            executor=self.executor,
            memory_store=sample.memory_store,
            query=sample.query,
            question_image=sample.metadata.get("question_image"),
        )
        corrections: list[StepCorrection] = []
        actions = list(student_actions)[: self.max_steps]
        for step_index, student_action in enumerate(actions):
            if session.stopped:
                break
            history = list(session.trace)
            observation = session.public_state()
            feedback = self._feedback(sample, session) if self.include_feedback else {}
            teacher_policy = self.teacher.correct_next(
                query=sample.query,
                gold_answer=sample.gold_answer,
                history=history,
                observation=observation,
                feedback=feedback,
                privileged_context=sample.metadata.get("teacher_privileged_context"),
            )
            teacher_actions, raw_response = self._normalize_teacher_policy(teacher_policy)
            example = self._sft_example(
                sample=sample,
                step_index=step_index,
                round_index=round_index,
                history=history,
                observation=observation,
                feedback=feedback,
                teacher_actions=teacher_actions,
                student_next_action=student_action,
                teacher_raw_response=raw_response,
            )
            corrections.append(
                StepCorrection(
                    sample_id=sample.sample_id,
                    step_index=step_index,
                    history=history,
                    observation=observation,
                    feedback=feedback,
                    teacher_actions=teacher_actions,
                    example=example,
                    student_next_action=student_action,
                    teacher_raw_response=raw_response,
                )
            )
            session.execute(student_action)
        return corrections

    def _feedback(self, sample: OPDSample, session: OPDToolSession) -> dict[str, Any]:
        if not session.evidence:
            return {
                "correct": False,
                "score": 0.0,
                "reason": "No retrieved evidence is available at this state.",
                "evidence_count": 0,
                "recommended_tool": "RETRIEVE",
            }
        if self.answer_model is None or self.judge is None:
            return {
                "correct": False,
                "score": 0.0,
                "reason": "No answer verifier configured; evidence is unverified.",
                "evidence_count": len(session.evidence),
            }
        try:
            prediction = self.answer_model.answer(
                query=sample.query,
                evidence=session.evidence,
                question_image=sample.metadata.get("question_image"),
            )
            correct, score, reason = self.judge.evaluate(sample.query, prediction, sample.gold_answer)
            return {
                "correct": bool(correct),
                "score": float(score),
                "reason": reason,
                "prediction": prediction,
                "evidence_count": len(session.evidence),
            }
        except Exception as exc:
            return {
                "correct": False,
                "score": 0.0,
                "reason": f"verification_error: {exc}",
                "evidence_count": len(session.evidence),
            }

    @staticmethod
    def _normalize_teacher_policy(policy: PolicyOutput | list[ToolAction]) -> tuple[list[ToolAction], str]:
        if isinstance(policy, PolicyOutput):
            return list(policy.actions), policy.raw_response
        return list(policy), ""

    def _sft_example(
        self,
        *,
        sample: OPDSample,
        step_index: int,
        round_index: int,
        history: list[ToolAction],
        observation: dict[str, Any],
        feedback: dict[str, Any],
        teacher_actions: list[ToolAction],
        student_next_action: ToolAction,
        teacher_raw_response: str,
    ) -> SFTExample:
        return SFTExample(
            sample_id=f"{sample.sample_id}:step:{step_index}",
            input=build_step_student_prompt(
                query=sample.query,
                history=history,
                observation=observation,
                schema=self.validator.schema_text(),
            ),
            target=json.dumps(
                [action.to_dict() for action in teacher_actions],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            round_index=round_index,
            metadata=self._sft_metadata(
                step_index=step_index,
                student_next_action=student_next_action,
                feedback=feedback,
                teacher_raw_response=teacher_raw_response,
            ),
        )

    @staticmethod
    def _sft_metadata(
        *,
        step_index: int,
        student_next_action: ToolAction,
        feedback: dict[str, Any],
        teacher_raw_response: str,
    ) -> dict[str, Any]:
        metadata = {
            "opd": {
                "mode": "step_level_correction",
                "step_index": step_index,
                "student_next_action": student_next_action.to_dict(),
                "teacher_raw_response": teacher_raw_response,
            }
        }
        if feedback:
            metadata["opd"]["teacher_feedback"] = feedback
        return metadata
