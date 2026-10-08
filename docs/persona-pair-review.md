# Persona trait interplay: review of all trait pairs

*2 October 2026. Traits, levels and anchors: `triagesim/personas.py`; evidence:
`docs/persona-traits-review.md`.*

Personas are every combination of trait levels, so two levels can only be combined if one person
can show both at once. Each pair of traits (55 patient, 21 nurse) was reviewed level by level
against the anchors. A pair is **incompatible** only if the two anchors contradict each other. A
**tension** (one level makes the other less common or harder) is kept: such patients and nurses are
realistic and worth simulating. Where two anchors overlapped by accident (one trait's anchor
describing another trait's dimension), the wording was fixed instead of excluding combinations.

An earlier attempt to have the dialogue master (Gemma-4-E4B) judge the 654 level pairs was unstable
and is not used: depending on the wording of its instruction it flagged 16 pairs (14 of them
tensions) or none (missing both contradictions below).

## Incompatible (excluded in `personas.INCOMPATIBLE`)

| Pair | Why |
|---|---|
| `verbosity`=terse + `focus`=frequent_tangents | "Answers in a few words" vs "often drifts into side stories before answering": a side story cannot fit in a few words. Off-target talk is itself a form of verbosity (Arbuckle 1993). |
| `expertise`=novice + `algorithm_reliance`=intuition_led | "Asks about each history topic in turn" vs "follows the patient's story rather than the algorithm's order" are opposite orders for the interview; novices work by rules and intuition develops with experience (Benner 1982; Cioffi 1998). |

Effect: 13,122 of 118,098 patient combinations (1/9) and 243 of 2,187 nurse combinations (1/9)
are excluded, leaving 104,976 patient and 3,888 nurse personas (nurses for each gender).

## Patient pairs

- **`age_group` × every other trait (10 pairs): compatible.** Age band describes no behaviour. It
  shifts likelihoods (older patients more often have patchy recall or tangents: Chung 2002;
  Arbuckle 1993), which is a sampling weight, not a contradiction.
- **`verbosity` × `focus`:** terse × frequent_tangents incompatible (above). Terse ×
  occasional_tangents is compatible: brief unrelated detail fits in a few words ("Yesterday, after
  bingo."). Typical or expansive × any focus level: compatible.
- **`health_literacy` × `recall_reliability`: compatible after rewording.** Adequate literacy
  read "names their conditions and medicines", which overlapped with recall ("recalls few of their
  medicines"). Literacy now means knowing the names; recall means remembering the facts (reliable
  recall: "by name or description").
- **`english_proficiency` × `health_literacy`: compatible.** Language vs health knowledge; they
  are correlated, not contradictory.
- **`english_proficiency` × `verbosity`, × `disfluency`, × `symptom_reporting`: compatible.**
  Fragmentary answers can be short or longer, need not contain fillers, and can minimise ("small
  pain, okay") or amplify ("very very bad").
- **`symptom_reporting` × `distress`: compatible by design.** The review split pain expression
  into these two so that a patient can play pain down while frightened, or report it high while
  composed.
- **`symptom_reporting` × `verbosity`: compatible** ("Worst pain ever." is terse and amplifying).
- **`distress` × `demeanour`, × `verbosity`: compatible.** Fear can show as impatience or
  hostility; distress can be brief ("Please, I'm scared.").
- **`demeanour` × `verbosity`, × `focus`: compatible after rewording.** Impatient read "is curt",
  which is a length; it now reads "presses to be seen sooner and shows irritation at waiting".
- **`english_proficiency` × `disfluency`: compatible after rewording.** Low disfluency read
  "speaks fluently", which could be read as language fluency; it now reads "rarely uses fillers or
  self-corrections".
- **All other patient pairs: compatible.** Their anchors describe separate dimensions: memory
  (recall), honesty about sensitive topics (disclosure), severity (reporting), emotion (distress),
  stance (demeanour), length (verbosity), topic (focus), fillers (disfluency), language (English)
  and health knowledge (literacy).

## Nurse pairs

- **`expertise` × `algorithm_reliance`:** novice × intuition_led incompatible (above). Novice ×
  blended and expert × protocol_driven are tensions, kept.
- **`expertise` × `risk_attitude`: compatible after rewording.** Novice read "covers every topic
  systematically", which could be read as wide danger screening (risk attitude's dimension); it now
  reads "asks about each history topic in turn rather than prioritising". An expert who screens the
  key red flags early but little beyond (risk_accepting) is coherent.
- **`expertise` × `questioning_style`: compatible.** Novice no longer reads "checklist-like"
  (question form); an expert can open with a general question and then ask targeted ones.
- **`workload` × `emotional_responsiveness`, × `risk_attitude`, × `questioning_style`, ×
  `expertise`: compatible after rewording.** Overloaded read "a short interview, quick closure,
  little time for reassurance", overlapping with empathy and screening; workload now describes pace
  only ("is rushed and moves on quickly"). A rushed nurse can still be empathic or screen widely,
  only faster. Lower empathy and earlier closure under load are tendencies in the literature, not
  rules.
- **`self_report_credence` × `questioning_style`: compatible.** Probing reported severity can use
  closed questions ("Can you walk on it?").
- **All other nurse pairs: compatible.** Their anchors describe separate dimensions: experience
  (expertise), interview order (algorithm reliance), breadth of danger screening (risk attitude),
  treatment of self-report (credence), question form (questioning style), response to emotion
  (emotional responsiveness) and pace (workload).
