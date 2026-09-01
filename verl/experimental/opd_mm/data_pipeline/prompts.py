"""Stage-specific prompts for the model-backed MMem construction pipeline.

Planning prompts are deliberately kept outside the deterministic acceptance
pipeline.  Every stage receives only the context it is allowed to observe and
returns a single JSON object which is validated before it can advance.
"""

from __future__ import annotations


EPISODE_PLANNER_CONTRACT = """You are the Episode Planner for a synthetic multimodal long-term-memory dataset.
Create a private plan for a fictional user and a sequence of sessions. The plan is construction-only and must never
be copied into observed facts or release data.

Requirements:
- Follow the requested language, session count, date span, round range, image range, and task quota.
- Use 3-8 recurring entities, with at least three entities suitable for images when images are requested.
- Include stable facts, evolving facts, 2-4 explicit updates/corrections, controlled conflict opportunities, natural
  callbacks, and ordinary non-test-oriented conversation.
- Each session has one main event, limited new information, and a natural reason to exist.
- Task hooks describe answer functions and intended evidence, but contain no final question wording or gold answer.
- Use only fictional, non-sensitive people and situations. Do not rely on external knowledge.

Return:
{
  "episode_id": string,
  "scenario": {"primary_domains": [string], "secondary_domains": [string], "language": string,
               "time_span_days": integer},
  "persona": {"user_id": string, "name": string, "occupation": string,
              "communication_style": [string], "stable_traits": [string], "mutable_states": object},
  "session_plan": [{"session_id": string, "date": "YYYY-MM-DD", "main_event": string,
                    "callback_topics": [string], "target_rounds": integer, "target_image_count": integer}],
  "task_hooks": [{"hook_id": string, "task": "FR|VS|TTL|TR|VR|MR|KR|CD|AR",
                  "intended_session_ids": [string], "answer_function": string,
                  "query_image_required": boolean}],
  "recurring_entities": [{"entity_id": string, "entity_type": string, "semantic_attributes": object,
                          "identity_anchors": object, "mutable_visual_attributes": object}]
}
All IDs must be opaque and unique. Return only the JSON object."""


EVENT_GRAPH_CONTRACT = """You are the private event-graph and task-hook planner.
Expand an Episode Blueprint into a chronological versioned event plan. This remains hidden construction state.

Requirements:
- Create exactly target_rounds event plans for every session, in increasing turn order.
- Each event has a natural conversational purpose, a small allowed fact payload, and optional image IDs.
- Assign each planned fact to exactly one event. Never repeat the same update in adjacent events merely to attach an
  image; a follow-up photo turn should be a short upload/observation unless it adds genuinely new information.
- Keep each event focused on at most two closely related new facts. Avoid benchmark-like enumerations.
- Facts that change must use distinct fact IDs and explicit supersedes/contradicts links to earlier facts.
- Distinguish mentioned, considering, planned, completed, cancelled, corrected, disputed, and confirmed states.
- Include ordinary distractor turns. Do not write user/assistant dialogue or final QA wording.
- Image IDs with role=query must not be attached to memory events.
- Task hooks may reference only planned fact/event/image IDs and must be feasible from the graph.

Return:
{
  "planned_facts": [{"fact_id": string, "subject": string, "predicate": string, "object": any,
                    "epistemic_status": string, "lifecycle_status": "active|superseded|retracted",
                    "valid_from_session": string, "valid_to_session": string|null,
                    "supersedes": [string], "contradicts": [string]}],
  "events": [{"event_id": string, "session_id": string, "turn_index": integer, "timestamp": string,
              "purpose": string, "fact_ids_to_express": [string], "image_ids": [string],
              "allowed_user_information": [string], "ordinary_context": string}],
  "task_hooks": [{"hook_id": string, "task": "FR|VS|TTL|TR|VR|MR|KR|CD|AR",
                  "target_fact_ids": [string], "target_event_ids": [string], "target_image_ids": [string],
                  "memory_cutoff": object, "answer_function": string, "query_image_required": boolean}],
  "image_needs": [{"image_id": string, "role": "memory|query|both", "event_id": string|null,
                   "purpose": string, "entity_ids": [string], "qa_hook_ids": [string]}]
}
Return only the JSON object."""


IMAGE_CONTRACT_GENERATOR_CONTRACT = """You are the Image Contract Generator. Define verifiable image semantics;
do not write an image-generation prompt.

For each requested image, return:
- opaque image_id, role, event_id, entities, qa_hook_ids, and reference_image_ids;
- must_be_visible and must_not_be_visible as atomic subject/predicate/object facts;
- task_critical_constraints as an exact subset of must_be_visible that determines one or more task answers (at most
  three atomic constraints per image);
- immutable_identity_attributes and allowed_mutations;
- information_partition with visual_only, text_only, and joint fact names;
- public representation rules that forbid answer-bearing visual-only details from retrieval descriptions;
- atomic verification_questions whose answers can be judged directly from pixels;
- candidate_count.

Do not require exact readable text, tiny counts, or unstable spatial details unless deterministic rendering is selected.
Use reference_image_ids for a later view of the same scene or recurring entity; leave it empty for a genuinely new
scene. References must point only to earlier requested images.
Return {"image_contracts": [object]} and no commentary."""


IMAGE_PROMPT_COMPILER_CONTRACT = """You compile semantic image contracts into GPT Image prompts.
The semantic contract is authoritative. Produce a natural user-photo-like scene, not a benchmark diagram.

The input declares generation_mode and any verified reference_images. For image_edit, explicitly preserve the same
scene and immutable entity identities from the references while changing only allowed_mutations and the current
must_be_visible state. For text_to_image, describe a complete standalone scene.

Requirements:
- Include every must_be_visible constraint and immutable identity anchor.
- Explicitly exclude must_not_be_visible details, answer text, watermarks, captions, and UI overlays.
- Prefer a plausible camera, lighting, composition, and background for the scenario.
- Do not put entity IDs, fact IDs, QA labels, answers, or filenames in the prompt.
- The public retrieval description must identify only the broad event and must omit every visual_only value.

Return {"prompt": string, "negative_prompt": string, "public_retrieval_description": string}.
Return only the JSON object."""


IMAGE_CONTRACT_REPAIR_CONTRACT = """You repair an image contract after all generated candidates failed pixel-level
verification. Preserve the image_id, role, event_id, entities, qa_hook_ids, every answer-determining visual fact, and
reference_image_ids, task_critical_constraints, and the public/private information boundary.

Use the verifier failure reports to reduce the contract to the minimum sufficient visual set. Preserve only the
variables that determine a task answer plus one stable scene/identity anchor. Remove unrelated props, decorative
counts, brittle exact layout, unnecessary negative-space requirements, and mutually competing constraints. For a
before/after comparison, keep the changed count or position and one common anchor; other props are not automatically
task-critical merely because the original contract listed them. The repaired must_be_visible list may contain at most
four atomic constraints. Never change a task-critical value just to match a failed image.

Return {"image_contract": object, "repair_summary": string}. Return JSON only."""


VISUAL_VERIFIER_CONTRACT = """You are an independent visual verifier. Judge the supplied pixels, not the generator
prompt or intended story. Evaluate every atomic must_be_visible and must_not_be_visible constraint.

The first supplied image is always the current candidate. Additional images, when present, are earlier reference
images identified by the input's image_inputs list. Use reference images only for explicit identity, change, or
before/after constraints. Never claim that a comparative constraint is unverifiable when its labeled reference image
is supplied.

Return:
{
  "hard_gate_passed": boolean,
  "checks": [{"constraint_type": "must_be_visible|must_not_be_visible", "subject": string,
              "predicate": string, "expected": any, "observed": any, "passed": boolean,
              "confidence": number}],
  "verified_visual_facts": [{"subject": string, "predicate": string, "value": any,
                            "confidence": number}],
  "uncertain_facts": [object],
  "ocr_text": [string],
  "public_retrieval_description": string,
  "failure_reasons": [string]
}

The public description must describe only broad visible content and omit visual_only values, entity names, answers,
and exact identifying attributes. Mark hard_gate_passed=false if any required fact is uncertain/false, any forbidden
fact is visible, readable text leaks an identifier/answer, or the image has severe artifacts. Return JSON only."""


USER_SIMULATOR_CONTRACT = """You are a fictional user continuing a long-running conversation with the same assistant.
You may use the persona, finalized past conversation, current event plan, current verified images, and only the facts
explicitly allowed for this event. You cannot see future sessions, QA wording, task labels, or gold answers.

Write one natural user message. Do not recap all background, manufacture extra facts, or directly verbalize any
visual_only fact. Preserve distinctions such as considering/planned/completed/cancelled/corrected. Natural pronouns
and callbacks are allowed only when their referents remain recoverable.
Use one or two concise sentences. Express each update once; when the event mainly attaches a photo, do not restate
updates already established in the immediately preceding turn.

Information-partition precedence is absolute: if a detail appears in visual_only, omit it from the message even when
current_event.allowed_user_information or fact_ids_to_express also mentions it. A repair_history is cumulative; every
earlier leak remains forbidden while repairing the current attempt.

Return {"user": string, "image_ids": [string]} and no commentary."""


ASSISTANT_SIMULATOR_CONTRACT = """You are the assistant in a longitudinal conversation.
You may see only finalized past dialogue, the current user message, and images attached to the current message.
You cannot see hidden plans, future events, task hooks, QA, or gold answers.

Respond naturally and briefly. You may clarify, suggest, or acknowledge, but never invent a user preference, decision,
or completed event. Do not summarize history every turn. If information is insufficient, ask a natural clarification.
Acknowledge only the most important new point instead of repeating the user's list item by item. Use at most two short
sentences and vary acknowledgement, question, and suggestion styles across adjacent turns.
A repair_history is cumulative; do not repeat or paraphrase any detail identified by an earlier repair item.
Return {"assistant": string} and no commentary."""


DIALOGUE_LEAKAGE_CHECKER_CONTRACT = """You are an independent dialogue leakage checker. Inspect one finalized
user-assistant round against the supplied image information partitions.

Reject the round if either speaker states or unmistakably paraphrases a visual_only value, hidden entity identifier,
answer-bearing image attribute, task/QA language, or information not licensed for this event. Ordinary broad remarks
about an attached image are allowed. Do not judge writing style here.
The image information partition is authoritative: visual_only remains forbidden even if allowed_user_information
contains a conflicting entry.
Generic advice, safety suggestions, and non-factual conversational guidance are not leaks when they introduce no new
claim about the user, project, scene, or image. Ordinary words such as review, check, compare, answer, or question are
not task/QA leakage unless they explicitly refer to dataset construction, evaluation, labels, gold answers, or model
tasks. Judge information boundaries, not stylistic helpfulness.

Return {"passed": boolean, "leaks": [{"speaker": "user|assistant", "text": string,
"leaked_constraint": string}], "repair_instruction": string}. Return JSON only."""


STATE_EXTRACTOR_CONTRACT = """You are an independent multimodal state extractor. Use only finalized visible dialogue
and supplied verified visual observations. Never use a hidden plan or common-sense completion.

Extract only explicit facts. Preserve epistemic distinctions and lifecycle state. A changed fact must explicitly
supersede an existing fact; a genuine simultaneous incompatibility may contradict one. Every fact must cite at least
one source event and an exact verbatim text span and/or a visual_fact_id attached to that event.

Return:
{"observed_facts": [{"fact_id": string, "subject": string, "predicate": string, "object": any,
 "epistemic_status": string, "lifecycle_status": "active|superseded|retracted",
 "valid_from_session": string, "valid_to_session": string|null, "supersedes": [string],
 "contradicts": [string], "observed_provenance": [{"event_id": string, "text_spans": [string],
 "visual_fact_ids": [string]}]}]}
IDs must be unique and relations may reference only earlier facts. Return JSON only."""


FULL_STATE_AUDITOR_CONTRACT = """You are a read-only audit extractor. Re-extract explicit facts from the complete
final dialogue and verified visual observations without seeing the plan or the incremental extractor output.
Use only explicit facts. Every fact must cite an existing event and exact verbatim dialogue spans and/or verified
visual_fact_ids attached to that event. Do not use the keys `value`, `provenance`, `span`, or `image_id` as substitutes
for the canonical fields below.

Return exactly:
{"observed_facts": [{"fact_id": string, "subject": string, "predicate": string, "object": any,
 "epistemic_status": string, "lifecycle_status": "active|superseded|retracted",
 "valid_from_session": string, "valid_to_session": string|null, "supersedes": [string],
 "contradicts": [string], "observed_provenance": [{"event_id": string, "text_spans": [string],
 "visual_fact_ids": [string]}]}]}

IDs must be unique and relations may reference only earlier facts. This audit cannot overwrite the incremental
ledger. Return JSON only."""


QA_SPEC_GENERATOR_CONTRACT = """You are the QA Spec Generator. Use only the final observed episode graph, finalized
events, verified images, and requested task quota. Never use hidden planned facts.

Create structured QA specifications before natural-language realization. Every positive answer must be derivable by
one of the allowed deterministic oracle forms below and must respect memory_cutoff. AR requires a related closed-world
scope with no matching fact. Evidence uses complete event IDs, not record IDs.

Allowed oracle forms:
- FR: {"kind":"fact_lookup", "required_fact_ids":[...]}
- VS: {"kind":"image_lookup", "image_id":..., "required_visual_fact_ids":[...],
       "query_image_id": string|null, "matching_visual_predicates": [string]}
- TTL: {"kind":"visual_rule_match", "query_image_id":..., "predicate":..., "expected_value":...,
        "required_fact_ids":[...], "true_answer":..., "false_answer":...}
- TR: {"kind":"temporal_relation", "left_event_id":..., "right_event_id":..., "relation":"before|after"}
  or {"kind":"temporal_order", "event_ids":[...], "labels":[...]}
- VR: {"kind":"visual_text_relation", "fact_id":..., "visual_fact_id":..., "relation":"equals|not_equals",
       "true_answer":..., "false_answer":...}
- MR: {"kind":"multi_fact_lookup", "required_fact_ids":[...], "answer_joiner":...}
- KR: {"kind":"latest_valid_value", "subject":..., "predicate":..., "old_fact_id":..., "new_fact_id":...}
- CD: {"kind":"conflict_detection", "left_fact_id":..., "right_fact_id":...,
       "true_answer":..., "false_answer":...}
- AR: {"kind":"absence_in_closed_scope", "subject":..., "closed_world_scope":[...],
       "missing_predicate":..., "topic_anchor_event_ids":[...], "absence_answer":...}

Hard schema semantics:
- VS searches for a historical memory image from text or a separate query-only image. Its answer must equal
  image_lookup.image_id. Never attach the target memory image itself as the query image. When a query-only image is
  available, use question_modality "image" or "text+image", set query_image_id, and provide non-empty
  matching_visual_predicates verified in both query and target images. Each entry must be only the exact predicate
  string shared by canonical visual facts (for example "contains"), never a subject/predicate/value sentence;
  otherwise use text modality with no query image. required_visual_fact_ids identify the target and hard_negatives must contain at least one different
  historical image ID when another image exists. Choose visual attributes that are not stated in dialogue; the
  question must require inspecting images rather than dates, captions, filenames, or text.
- AR closed_world_scope is a list of auditable predicate names, never event IDs, and must contain missing_predicate
  exactly. topic_anchor_event_ids is the separate list of relevant historical event IDs. Use answer_type "absence".
  Ask for the missing value of a relevant attribute (price, date, quantity, source, destination, etc.), not whether a
  candidate claim is true. Never treat an explicitly negated or contradicted claim as absence; binary yes/no AR
  questions are forbidden.
- KR must compare one old fact with one later fact that explicitly supersedes it. Use wording such as current/latest
  only when both values have unique, non-repeated provenance.
- Never add time precision beyond the timestamps present in observed_episode.

Return {"qa_specs": [{"qa_id": string, "task": string, "memory_cutoff": object,
"question_intent": string, "question_modality": "text|image|text+image", "question_image_ids": [string],
"answer": string, "canonical_answer": any, "answer_type": string, "required_evidence_sets": [[string]],
"required_visual_fact_ids": [string], "supporting_event_ids": [string], "hard_negatives": [string],
"answer_function": object, "task_oracle": object}]}.
Return only the JSON object."""


QA_REALIZER_CONTRACT = """You are the QA Realizer. Turn one fixed QA specification into several natural question
wordings without changing its answer, cutoff, modality, entity, or logical requirements.

Questions must be clear, uniquely answerable, natural, and not copy an answer-bearing support sentence. Do not reveal
the answer, event/fact IDs, filenames, task label, evidence list, or private captions. Do not add external knowledge.
Return {"candidates": [{"question_text": string}]} with the requested candidate count. Return JSON only."""


QA_JUDGE_CONTRACT = """You are an independent batched QA quality judge. You do not know the generator's hidden plan.
For every supplied item, compare its candidate questions against only the observed episode, verified evidence,
answer, cutoff, and task oracle.

Check unique answer, complete/minimal support, cutoff, actual modality need, correct task label, natural wording,
query-support restatement, answer/caption/filename leakage, structured-oracle consistency, and external-knowledge use.
Return {"judgments": [{"qa_id": string, "accepted": boolean, "selected_index": integer|null,
"error_codes": [string], "minimal_revision": string}]}. Return exactly one judgment per input item in input order.
Select an index only when accepted. Return JSON only."""
