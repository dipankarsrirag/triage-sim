# Grounding TriageSim persona traits in ED triage research

*Proposal, 1 October 2026. Scope: the patient and nurse persona fields in `triagesim/personas.py`.
Ethnicity and country of origin are out of scope (TTS only). Every reference was checked against
PubMed, Crossref, arXiv, the ACL Anthology or the issuing body's website. Figures are quoted from
abstracts or the primary documents.*

## Decisions (1 October 2026)

Implemented in `triagesim/personas.py`, which is now the single source of truth for traits, levels,
behavioural anchors and sources:

1. **Nurse traits shape the interview only.** Risk attitude, algorithm reliance and self-report
   credence are recast as interview behaviours (e.g. a risk-averse nurse screens more widely for red
   flags). They must not cause mistriage: the nurse is told its persona does not change how it
   applies the algorithm or the level it assigns.
2. **Reporting traits and the agreement filter (documented, not resolved).** Evaluation pools keep
   only episodes whose nurse reached the ground-truth level. Patients with minimising or amplifying
   `symptom_reporting`, or selective or guarded `disclosure`, mislead the nurse more often, so their
   episodes are filtered out more often and the pool under-represents them. Report the persona mix
   of every pool. This will be revisited with the game-theoretic view of triage (§6, item 7).
3. **Case-dependent states come from the seed.** `cognitive_state` and `intoxication` are not
   sampled: the dialogue master sets them in the patient's script from the case.
4. **Personas are generated combinatorially.** Every combination of levels is enumerated, except
   pairs of levels that contradict each other by definition. All 76 trait pairs were reviewed level
   by level (`docs/persona-pair-review.md`): two pairs are excluded (`personas.INCOMPATIBLE`), and
   anchors that overlapped another trait's dimension were reworded so traits combine freely. Gender
   comes from the case; age group is sampled until the cases carry age. Fabrication (§3, recall) is
   left out for now.

## 0. Summary

**Patient.** Some fields come from the case: `age_group`, `gender`, optional `prior_ed_use`. Two states
depend on the case: `cognitive_state` and `intoxication`. The sampled traits are `english_proficiency`,
`health_literacy`, `recall_reliability`, `symptom_reporting`, `distress`, `disclosure`, `demeanour`,
`verbosity`, `focus` and `disfluency`.

**Nurse.** The traits are `expertise`, `algorithm_reliance`, `risk_attitude`, `self_report_credence`,
`questioning_style` and `emotional_responsiveness`. `workload` is set per episode. `gender` is used
only for the nurse's name and for TTS.

Five changes matter most. Each is argued in §1–4:
1. **Case signs are not persona traits.** Confusion, intoxication, severe pain and distress are triage
   criteria in ESI and ATS. If they are sampled independently of the case, they silently change the
   correct triage level.
2. **Split `pain_expression`.** It becomes a signed reporting bias against the case pain score
   (`symptom_reporting`) plus emotional display (`distress`). ESI and ATS map *reported* pain straight
   onto acuity, and clinicians routinely discount pain.
3. **Recast nurse experience as interview style, not accuracy.** Knowledge predicts triage accuracy;
   experience does not.
4. **Replace vague style fields with coded behaviours.** Nurse `communication_style`, `verbosity` and
   `emotional_expression` become open vs closed questioning and response to emotional cues, both
   drawn from conversation-analysis, RIAS and VR-CoDES research. Two new fields are added:
   `health_literacy` in place of `socioeconomic_status`, and `disclosure` in place of
   `trust_in_healthcare`.
5. **Tag each trait with how it can be verified.** The per-turn judge can only enforce U and G traits
   (§1). Risk attitude and algorithm reliance show up only at episode level.

## 1. Design principles from the evidence

- **P1. Keep persona separate from presentation.** ESI v5 counts newly altered mental status and
  severe pain or distress as level-2 criteria (ENA 2023). ATS descriptors grade "very severe",
  "moderately severe" and "moderate" pain, and behavioural disturbance, across categories 2–4
  (ACEM 2023a). A persona that is "confused" or "in agony" therefore carries clinical information and
  must agree with the case's acuity and diagnoses.
- **P2. Three levels of verifiability.**
  - **U:** checkable from the utterance and the dialogue.
  - **G:** checkable only against the case or script, e.g. reported vs recorded pain.
  - **E:** visible only across a whole episode or many episodes, e.g. over-triage rate.

  The judge in `prompts.judge_messages` runs per turn, so it can enforce U and G traits only. E traits
  need aggregate metrics.
- **P3. Anchor every level in behaviour.** Levels should be defined by coded behaviours, as in RIAS
  (Roter 2002) and VR-CoDES (Zimmermann 2011), not by "low/medium/high". LLM simulators whose traits
  proved recoverable by judges used anchored categorical levels: PatientSim with 2–6 per axis
  (Kyung 2025) and VeriSim with a 0–4 scale (Mansouri 2026).
- **P4. Sample traits together, weighted by prevalence.** ED traits co-occur: older age with cognitive
  impairment and poorer recall, limited English proficiency with limited health literacy,
  intoxication with young men. Sampling each trait independently and uniformly over-represents rare
  states and produces implausible combinations.
- **P5. Keep traits apart from goals.** Non-strategic tendencies such as catastrophizing or stoicism
  must stay separable from the planned strategic patient goal. Otherwise the two effects cannot be
  told apart.

## 2. Evidence summary

### 2.1 How patients communicate and present

- **Health literacy.**
  - About 40% of US ED patients have health literacy at or below 8th-grade level (Herndon 2011).
  - 59% of Australians aged 15–74 scored below Level 3 on the health literacy scale (ABS 2006).
  - Older ED patients named 78% of their prescription medicines, but only 43% named all of them
    (Chung 2002).
  - More educated patients take a more active part in consultations (Street 2005).
- **Limited English proficiency (LEP).**
  - LEP patients receive less explanation and are less satisfied. Professional interpreters are
    under-used in EDs, and ad hoc interpreters make more errors (Flores 2005; Ramirez 2008).
  - In Queensland EDs, 19–23% of Vietnamese, Chinese and Arabic speakers needed an interpreter, and
    home language predicted longer length of stay (Mahmoud 2013).
  - In Australia, 3.4% of the population speaks English "not well or not at all", and 22.8% use
    another language at home (ABS 2022).
- **Pain.**
  - Pain is the chief complaint in 52% of ED visits (Cordell 2002).
  - Triage scales use self-reported pain directly. ESI v5: patients with pain ≥7/10 "should be
    considered" for level 2, judged "by patient report, corroborated with clinical observation"
    (ENA 2023). ATS grades pain across categories 2–4 (ACEM 2023a).
  - Clinicians rate pain lower than patients do. At triage, nurses gave 5.1 against patients' 7.5
    (Puntillo 2003). ED physicians were 1.3 points lower, and experts more so than novices
    (Marquié 2003).
  - Italian triage nurses say they adjust patients' pain scores because they believe patients
    exaggerate to get a higher priority. The numeric rating scale (NRS) also yielded higher scores
    than a disguised scale (Capponi 2016).
  - Reporting style varies with:
    - catastrophizing (Sullivan 1995, 2001);
    - stoicism, which predicted self-rated pain better than age did (Yong 2006);
    - somatosensory amplification (Barsky 1988);
    - the social encoding and decoding of pain (Craig 2009).
  - Patients under-ask: only 31% of those who wanted analgesia asked for it (Todd 2007).
- **Anxiety and distress.**
  - 24% of patients in an Australian ED had high or very high K10 distress (Fulbrook 2015).
  - 25% of ED chest-pain patients met criteria for panic disorder, and 98% of these went
    unrecognised (Fleet 1996).
  - Current anxiety and pain bias which symptoms patients recall (Barsky 2002).
  - VR-CoDES operationally defines emotional *cues* and *concerns* (Zimmermann 2011).
- **Cognitive impairment, delirium and intoxication.**
  - Among ED patients aged ≥70, 26% had impaired mental status: 10% delirium and 16% cognitive
    impairment without delirium. Most cases were undocumented (Hustey 2002).
  - Among ED patients aged ≥65, 8.3% were delirious. 92% of these were hypoactive, and 76% were
    missed (Han 2009).
  - The Confusion Assessment Method (CAM) features are inattention, disorganised thinking, altered
    consciousness and acute or fluctuating onset (Inouye 1990).
  - 9.5% of ED presentations in Australia and New Zealand were alcohol-related (4.9–15.2% by site).
    These patients were more often male, younger (median 37) and brought by ambulance or police
    (Egerton-Warburton 2018).
  - Intoxication masks critical illness: 1% of apparently uncomplicated intoxicated patients later
    needed critical care (Klein 2018).
- **Recall and disclosure.**
  - Patients under-report past events, merge similar episodes (telescoping) and falsely recall
    symptoms (Barsky 2002).
  - 61–81% of surveyed adults had withheld medically relevant information from clinicians, mainly
    to avoid judgement or embarrassment (Levy 2018).
  - Distrust varies with SES, gender and race (Armstrong 2007). Hall (2001) conceptualises trust in
    physicians.
  - These sources are general. ED studies are summarised at the end of this section.
- **Help-seeking and frequent attendance.**
  - Across studies, 37% of ED visits were judged non-urgent (Uscher-Pines 2013).
  - Patients account for why their problem deserves care ("doctorability"; Heritage 2006b).
  - At ED triage, companions present problems at greater length and argue that the visit is
    reasonable (Lee 2015).
  - Delay in seeking care for ACS and stroke is common and shaped by how patients interpret symptoms
    (Moser 2006).
  - Frequent users are 4.5–8% of ED patients but 21–28% of visits, and many have real medical needs
    (LaCalle 2010). They are commonly defined as more than 4 visits a year, with more psychiatric and
    alcohol presentations (Locker 2007).
  - "Classic" drug-seeking behaviours, such as reporting 10/10 pain (29% of visits), are only
    moderately frequent even among drug-seeking patients (Grover 2012).
- **Older adults.**
  - 24% of Australian ED presentations are aged ≥65 (AIHW 2026).
  - 28.6% of older ED patients presented atypically, most often without fever (Limpawattana 2016).
    MI presented without chest pain in 42% of women and 31% of men (Canto 2012).
  - ESI's sensitivity for older patients who needed a life-saving intervention was only 42–46%.
    This was linked to not following ESI and to misreading vital signs and high-risk situations
    (Platts-Mills 2010; Grossmann 2012).
  - Off-target verbosity increases with age (Arbuckle 1993). Disfluency is only slightly higher in
    older speakers but rises with planning difficulty (Bortfeld 2001).

#### ED evidence for recall and disclosure (added 1 October 2026)

**Recall.** ED studies measure how well patients recall their medicines and past history. No ED
study of how accurately patients report when symptoms began was found.
- **Medicines.**
  - A medication list taken at triage differed from an independent re-interview for 37% of 1,657
    patients (mean age 39). 27.9% had a medicine omitted, 9.8% had a stopped medicine listed, and 38%
    took an over-the-counter medicine that was not recorded (Mazer 2011).
  - Among admitted patients aged over 64, 56% of ED medication lists omitted a medicine and 80% had a
    dose or frequency error (Caglar 2011).
  - No patient's triage medication list, taken from recall, fully matched urine mass spectrometry
    (Kreshak 2015).
  - Only 31% of self-reported 48-hour drug histories matched a urine drug screen. 16 of 55 patients
    had taken a drug they did not report, and no demographic factor predicted accuracy (Monte 2015).
  - Older patients under-report medicine use (Rockett 2006) and name fewer of their medicines
    (Chung 2002).
  - These studies cannot fully separate the patient's recall from how the history was taken.
- **Past history and symptoms.**
  - In 776 Australian ED chest-pain patients, self-reported history agreed with cardiologists'
    adjudication substantially but not perfectly. κ ranged from 1.00 for previous bypass surgery to
    0.33 for previous ventricular dysrhythmia (Iliceto 2016).
  - In 1,000 Swedish ED chest-pain patients, the patient's own tablet history agreed with the
    physician's record less for chest-pain characteristics (κ 0.19–0.70) than for risk factors
    (κ 0.55–0.91) (Brandberg 2024).
- **Pattern.** Salient facts are recalled well. The errors are omissions, approximate details and
  answers that change when asked again. A patient can show all of these without inventing anything.

**Disclosure.** ED non-disclosure depends on the topic and on how and where the patient is asked.
- **Substances.**
  - In 1,502 Tennessee ED patients, toxicology screening barely changed estimated alcohol use but
    markedly raised estimated use of opioids, benzodiazepines, cannabis and amphetamines. Use of any
    of the eight substances rose from 44% to 56% (women) and from 61% to 69% (men). Patients aged ≥65
    had more undeclared use than those aged 18–24 (Rockett 2006).
  - Self-reported drinking was valid against measured blood alcohol in 10,741 injured ED patients
    from 16 countries (Cherpitel 2007).
  - In two Taiwanese EDs, about two-thirds of patients whose urine was positive for amphetamines or
    opiates had not reported use. Every reported use was confirmed (Chen 2006).
- **Violence and suicidality.**
  - In an urban US ED, 26% of women were at current risk of domestic violence on an exit
    questionnaire. Abuse was disclosed in 8% of usual-care encounters, and in 14% when a computer
    screen prompted the provider (Rhodes 2006).
  - Of 1,590 ED waiting-room patients with non-psychiatric complaints, 11.6% reported suicidal
    ideation on a computer screen. 25 of the 31 who had a plan were not detected during the visit
    (Claassen 2005).
  - Universal screening during routine ED care nearly doubled detection of suicide risk, from 2.9% to
    5.7% (Boudreaux 2016).
- **How the patient is asked.**
  - In audio-recorded ED visits, providers usually asked about domestic violence perfunctorily during
    the social history. Disclosure went with probing (at least one follow-up question), open-ended
    opportunities to talk, and responsiveness to the patient's cues (Rhodes 2007).
  - Walk-in triage patients disclosed hazardous drinking and high-risk drug use more often to a kiosk
    than to an interviewer, although 74% preferred the interviewer (Hankin 2015).
- **Reasons for withholding.**
  - In a Melbourne ED, 4% of patients changed or withheld information because others might overhear
    (Karro 2005). In Canadian rural and remote EDs, waiting rooms and triage areas drew the most
    privacy concerns (Geetha Manukumar 2025).
  - ED patients who knew a victim of domestic violence most often named two fears as barriers to
    disclosure: that the doctor would tell the police (31%) and that the perpetrator would find out
    (30%) (Kimberg 2021).
  - 97% of ED patients with opioid use disorder reported some fear of enacted stigma. Fear was lower
    when they felt more compassion from staff (Steinhauser 2024).
- **Pattern.** Patients deny or omit stigmatised topics but rarely alcohol. Disclosure increases when
  the clinician asks directly, follows up and offers privacy. Over-reporting was negligible
  (Chen 2006).

### 2.2 How triage nurses interview and decide

- **Experience.**
  - Benner (1982) describes five stages from novice to expert.
  - No study has found experience to be an independent predictor of triage accuracy. Factual
    knowledge is (Considine 2007).
  - Experienced nurses collect less data, rely on remembered cases and are more confident
    (Cioffi 1998).
  - Thinking strategies differ only slightly between high- and low-accuracy nurses (Göransson 2008).
  - Nurses describe a shift from structured to intuitive assessment as they gain experience
    (Gorick 2026).
- **Accuracy and risk.**
  - Agreement on vignettes is moderate:
    - ATS: 61% expected, 18% under-triage, 21% over-triage (Considine 2004).
    - CTAS: 57.6% concordance, κ 0.46 (Göransson 2005).
    - ETEK scenarios: κ 0.41 overall, 0.24 for mental health (Gerdtz 2008).

    Reviews rate accuracy as moderate and highly variable (Tam 2018; Zachariasse 2019).
  - In 5.3M ESI encounters, 28.9% were over-triaged and 3.3% under-triaged. Under-triage was more
    likely for Black patients (Sax 2023).
  - Clinician risk attitude changes decisions: risk-seeking physicians admitted 31% of chest-pain
    patients, against 53% for risk-avoiders (Pearson 1995).
  - Nurses cite fear of liability, intuition and gendered expectations (Arslanian-Engoren 2000).
- **Algorithm vs intuition.**
  - Triage nurses used objective physiological data sparingly (Gerdtz 2001).
  - Under uncertainty they rely more on the representativeness heuristic (Cioffi 1998).
  - Not applying ESI rules drives under-triage of older adults (Platts-Mills 2010; Grossmann 2012).
- **Interview structure.**
  - General-inquiry openings ("What can I do for you?") elicit longer presentations with more
    symptoms than confirmatory openings (Heritage 2006a).
  - Swedish telenurse calls later subject to malpractice claims had fewer open questions and fewer
    back-channels, and elicited less medical information (Ernesäter 2014).
  - Telenurses mostly ask closed questions and answer callers' concerns with more closed medical
    questions (Ernesäter 2016).
  - Either/or questions create interactional trouble (Erkelens 2021), and decision-support software
    drives the order of questions (Murdoch 2015).
  - Korean ED triage recordings studied with conversation analysis (CA) show that the presenter,
    patient or companion, shapes how problems are told (Lee 2015).
  - History-taking is the triage nurse's most frequent task (Considine 2026).
  - Australian ED discourse is shaped by time pressure, divergent goals and missing information
    about process (Slade 2008).
- **Patient-centredness and emotion.**
  - Mead (2000) sets out the dimensions of patient-centredness.
  - VR-CoDES-P codes clinician responses as explicit or non-explicit, and as providing or reducing
    space for the patient's emotion (Del Piccolo 2011).
  - Supportive talk increases patient participation (Street 2005).
  - Australian triage nurses endorse person-centred care but cite time and volume as barriers
    (Janerka 2025).
- **Workload.**
  - ACEM says triage "should take no more than two to five minutes" (ACEM 2023a, 2023b).
  - Triage duration varies with the nurse, the patient and the environment (Gerdtz 2001).
  - Simultaneous arrivals, crowding, interruptions and "triage fatigue" all degrade triage
    (Hitchcock 2014; Reay 2020; Fekonja 2023; Gorick 2026).

### 2.3 Standardised and LLM-simulated patients

Standardised-patient practice prizes reproducible, scripted portrayals (Barrows 1993; Lewis 2017).
Groves (1978) describes four "hateful patient" types. Physicians rated 15% of primary-care patients as
difficult, a group strongly associated with psychiatric disorder (Hahn 1996).

| Simulator | Patient axes | Validation |
|---|---|---|
| PatientSim (Kyung 2025; MIMIC-IV-ED) | Personality: neutral, impatient, overanxious, distrustful, overly positive, verbose. Language: CEFR A/B/C. Recall: low/high. Confusion: normal/high. 37 combinations. | LLM judge plus 4 clinicians, 4-point consistency rating |
| Patient-Ψ (Wang 2024) | Plain, upset, verbose, reserved, tangent, pleasing (from expert interviews) | 20 experts, 13 trainees |
| VeriSim (Mansouri 2026) | Recall, health literacy, emotion, style, non-disclosure, self-diagnosis; severity 0–4 | LLM judge κ = 0.77 against humans on noise fidelity |
| MedDialBench (Luo 2026) | Logic consistency, health cognition, expression, disclosure, attitude | Fabrication was 1.7–3.4× more damaging than withholding |
| PatientsWithPersonality (Schlager 2026) | HEXACO personality; selective disclosure | Rated "too informative" less often than baselines; traits recoverable |
| AgentClinic (Schmidgall 2024) | Cognitive bias (e.g. self-diagnosis) and implicit bias | Diagnostic accuracy and patient-perception metrics |

- CRAFT-MD recommends open-ended, conversational evaluation (Johri 2025).
- AMIE used validated patient actors (Tu 2025).
- EHR2Dial-Triage generates MIMIC-IV-ED triage dialogues (Zhu 2026).
- TriageSim's current axes closely mirror PatientSim's. The ED-specific axes proposed below are
  missing from both.

## 3. Proposed patient traits

Each trait lists observability (U/G/E, §1) and prevalence where it exists. ⚑ marks interaction with
the planned patient goal (§6).

**Case-derived context (not sampled freely).**
- `age_group` (child / adult / older ≥65):
  - Source: the case (MIMIC `anchor_age`, or scenario text).
  - It drives the correlated sampling below.
  - Prevalence: 24% of Australian ED presentations are aged ≥65 (AIHW 2026).
- `gender`: from the case, as now.
- `prior_ed_use` (none / occasional / frequent, i.e. >4 visits a year) ⚑:
  - Optional; take it from ED history where available.
  - Frequent users are 4.5–8% of ED patients (LaCalle 2010; Locker 2007).
  - Observability: U, when the patient mentions earlier visits.

**Case-conditioned states.** Sample these only when compatible with the case's complaint, diagnoses
and acuity (P1).
- `cognitive_state`: *alert* / *impaired* / *delirious*.
  - Impaired: slow, repeats itself, orientation lapses, needs questions repeated.
  - Delirious: CAM inattention plus disorganised thinking. Default to the hypoactive form: sparse,
    drowsy, loses the question.
  - Levels mirror the ED categories in Hustey (2002). Evidence: Han 2009; Inouye 1990.
  - Observability: U.
  - Weights for age ≥70: 0.74 / 0.16 / 0.10 (Hustey 2002). Delirium only when the case supports
    altered mental status.
- `intoxication`: *none* / *mild* (disinhibited, still coherent) / *marked* (slurred, repetitive,
  inconsistent).
  - Evidence: Egerton-Warburton 2018; Klein 2018.
  - Observability: U, helped by TTS.
  - Prevalence: 9.5% of presentations are alcohol-related, skewed male and younger. Only the "marked"
    level should move ATS behavioural descriptors.

**Communication capacity.**
- `english_proficiency` (renamed from `language_proficiency`): *fluent* / *functional* / *limited*.
  - Functional: simplified grammar, occasional misunderstanding.
  - Limited: fragments, frequent misunderstanding; an interpreter would normally be needed.
  - Levels map to the Census bands (very well / well / not well–not at all) and to PatientSim's CEFR
    A/B/C levels.
  - Evidence: Flores 2005; Ramirez 2008; Mahmoud 2013.
  - Observability: U.
  - Prevalence: limited is about 3.4% of the Australian population (ABS 2022). For admitted MIMIC
    patients, use the `admissions.language` field (MIT-LCP).
- `health_literacy` (new; replaces `socioeconomic_status`): *limited* / *marginal* / *adequate*.
  - Anchors:
    - lay versus technical vocabulary;
    - whether the patient can name medicines and conditions;
    - whether the patient can use a 0–10 scale;
    - whether the patient asks what jargon means.
  - Levels follow ABS Level 1 / Level 2 / Level ≥3 (ABS 2006).
  - Evidence: Herndon 2011; Chung 2002; Street 2005.
  - Observability: U for vocabulary and clarification requests; G for medicine names against the
    script.
  - Prevalence: limited literacy in about 40% of US ED patients.
- `recall_reliability` (redefines `recall_accuracy`; redefined again 1 October 2026 from the ED
  evidence in §2.1): *reliable* / *patchy* / *unreliable*.
  - Scope: memory for facts in the script, such as when symptoms began and how they progressed,
    current medicines and doses, and past conditions. Naming medicines and conditions belongs to
    `health_literacy`.
  - Reliable: gives clear timings and recalls all their current medicines and past conditions when
    asked.
  - Patchy: gives approximate timings ("sometime yesterday"). Recalls major conditions and regular
    medicines but forgets some, such as over-the-counter or occasional ones, and is unsure of doses.
  - Unreliable: cannot say when symptoms began or in what order. Recalls few of their medicines or
    past conditions, and gives a different time or detail when asked again.
  - Errors are omissions, approximations and changed answers about facts in the script. The patient
    never invents symptoms, medicines or history. **Fabrication** is out of scope (Decision 4): it
    breaks faithfulness to the case and is the most damaging behaviour for downstream models
    (Luo 2026).
  - Evidence (ED): Mazer 2011; Caglar 2011; Kreshak 2015; Monte 2015; Iliceto 2016; Brandberg 2024;
    Chung 2002. Barsky 2002 (general) describes the mechanisms: forgetting and telescoping.
  - Observability: G against the script. U for "I can't remember" and for an answer that changes
    when asked again.
  - Prevalence:
    - 63% of triage medication lists matched a re-interview (Mazer 2011; mean age 39).
    - 43% of older ED patients named all their medicines (Chung 2002).
    - This suggests a weight of about 0.6 for *reliable* in adults and about 0.4 at ≥65. No ED study
      separates *patchy* from *unreliable*.
  - Correlate with age and `cognitive_state`.

**Stance and affect.**
- `symptom_reporting` ⚑ (new; replaces the reporting half of `pain_expression`): *minimising* /
  *faithful* / *amplifying*.
  - It maps the case pain score and severity to what the patient reports.
  - Minimising: about 2 NRS points lower, "just a bit sore", benign explanations.
  - Amplifying: about 2 points higher (capped at 10), catastrophic wording.
  - Why a signed scale of about 2 points: triage thresholds are pain-based (ESI ≥7/10; ATS pain
    bands), and clinician–patient gaps run 1.3–2.4 points (Puntillo 2003; Marquié 2003).
  - Evidence: Barsky 1988; Sullivan 2001; Yong 2006; Moser 2006; Todd 2007; Capponi 2016.
  - Observability: G, reported against recorded pain. The judge must be told the shift is intended.
  - Prevalence: no ED estimate found. Weights are a design choice to report as such.
- `distress` ⚑ (merges `emotion_regulation`, `reactivity_to_clinician_emotion` and the display half
  of `pain_expression`): *calm* / *anxious* / *highly distressed*.
  - Anchors: how often VR-CoDES cues or concerns appear, reassurance-seeking, and whether distress
    settles after reassurance.
  - Evidence: Fleet 1996; Fulbrook 2015; Zimmermann 2011. ATS and ESI both name "distress".
  - Observability: U.
  - Prevalence: about 24% of ED patients at high or very high distress (Fulbrook 2015).
- `disclosure` ⚑ (redefines `trust_in_healthcare`; redefined again 1 October 2026 from the ED
  evidence in §2.1): *open* / *selective* / *guarded*.
  - Scope: sensitive topics in the script only. These are alcohol and drug use, mental health and
    self-harm, violence at home, and not taking medicines. Other history is unaffected, which keeps
    the trait apart from `verbosity` and `demeanour`.
  - Open: answers questions on sensitive topics truthfully when asked, but does not raise them
    unprompted. Volunteering everything is the known LLM over-sharing failure (Schlager 2026). In
    the ED, sensitive topics surface mainly when someone asks (Boudreaux 2016).
  - Selective: denies or plays down a sensitive topic when asked briefly or with a closed question
    (e.g. admits drinking but not drug use). Discloses it if the nurse follows up, asks openly,
    picks up a hint or reassures about privacy.
  - Guarded: denies or deflects sensitive topics even when the nurse follows up ("why do you need to
    know that?"), out of fear of judgement, the police or being overheard. Answers other questions
    normally.
  - The patient only withholds or plays down facts in the script and never invents any.
  - Evidence (ED): Rockett 2006; Chen 2006; Cherpitel 2007; Rhodes 2006, 2007; Claassen 2005;
    Boudreaux 2016; Hankin 2015; Karro 2005; Kimberg 2021; Steinhauser 2024. Levy 2018 (general)
    is background. Distrust is now one reason a patient may give, not a level, so Armstrong 2007
    and Hall 2001 are no longer needed for this trait.
  - Observability: G, comparing what was asked with what the script holds. The trait has no effect
    when the script holds no sensitive item. Sample it only for cases that carry one, e.g.
    alcohol- or drug-related, mental-health or injury presentations, or a social history in the case.
  - Interaction: whether a *selective* patient discloses depends on the nurse's `questioning_style`
    and `emotional_responsiveness` (Rhodes 2007). The trait therefore also probes how well the nurse
    interviews (§6, item 7).
  - Prevalence: among ED patients with something to disclose, non-disclosure in a routine encounter
    is common for:
    - illicit drugs: about two-thirds (Chen 2006);
    - domestic violence: 8% of women disclosed it in usual-care encounters, against 26% at risk
      (urban ED; Rhodes 2006);
    - suicidal ideation: 25 of 31 patients with a plan undetected (Claassen 2005).

    It is rare for alcohol (Cherpitel 2007; Rockett 2006). 4% of respondents in one Melbourne ED
    withheld information for privacy reasons (Karro 2005). No ED study separates *selective* from
    *guarded*. Weight by topic: mostly *open* for alcohol, and about two-thirds *selective* plus
    *guarded* for illicit drugs. Report the split as a design choice.
- `demeanour` ⚑ (new): *cooperative* / *impatient* (pushes to be seen, curt) / *hostile* (verbal
  aggression).
  - Evidence: Groves 1978; Hahn 1996; PatientSim "impatient"; Patient-Ψ "upset". ACEM (2023a) notes
    aggression risk at triage.
  - Observability: U.
  - Prevalence: about 15% "difficult" in primary care. No ED estimate found.

**Speech surface.**
- `verbosity` (kept): *terse* / *typical* / *expansive*.
  - Judge it relative to the question form: general inquiries legitimately elicit longer answers
    (Heritage 2006a).
  - Observability: U.
- `focus` (renamed from `topic_drift`): *on-target* / *occasional tangents* / *frequent tangents*.
  - Evidence: Arbuckle 1993; Patient-Ψ "tangent".
  - Observability: U. Correlate with age and cognitive state.
- `disfluency` (kept): *low* / *moderate* / *high*.
  - Specify fillers and repairs separately; they pattern differently (Bortfeld 2001).
  - Observability: U. Correlate with limited English, impairment and distress.

**TTS only (hidden from LLMs):** ethnicity or accent, and `instruction`. This matches the current
`HIDDEN_PERSONA_FIELDS`.

## 4. Proposed nurse traits

- `expertise` (redefines `experience_level`): *novice* / *competent* / *expert*.
  - Novice: exhaustive, checklist-like questions, recites algorithm steps, asks for vitals early.
  - Expert: hypothesis-driven, fewer questions, early red-flag screening.
  - Benner's five stages are collapsed to three because the triage studies contrast only less and
    more experienced nurses (Cioffi 1998; Gorick 2026).
  - **Do not tell experts to be more accurate** (Considine 2007). Experts may discount pain more
    (Marquié 2003, physicians).
  - Observability: U for style; E for number of questions.
  - Prevalence: real triage is done by trained, experienced staff (ACEM 2023a). An observed sample
    had a median of 5 years of triage experience, IQR 3.2–13 (Considine 2026).
- `algorithm_reliance` (redefines `guideline_adherence`): *protocol-driven* / *blended* /
  *intuition-led*.
  - Protocol-driven: names ESI or ATS criteria and gets vitals before deciding.
  - Intuition-led: decides from the gestalt of the presentation and may skip vitals.
  - Evidence: Gerdtz 2001; Cioffi 1998; Göransson 2008; Platts-Mills 2010; Grossmann 2012;
    Murdoch 2015.
  - Observability: E, from `check_vital` use and the reasoning trace.
- `risk_attitude` ⚑ (redefines `risk_tolerance`): *risk-averse* / *balanced* / *risk-accepting*.
  - Behaviour under uncertainty: whether the nurse rounds acuity up or down.
  - The three levels follow Pearson's (1995) low / medium / high risk-taking groups.
  - Evidence: Pearson 1995; Arslanian-Engoren 2000; Considine 2004.
  - Observability: E only, as over- and under-triage rates against ground truth for each persona.
  - Prevalence: real systems skew towards over-triage (28.9% over vs 3.3% under; Sax 2023).
- `self_report_credence` ⚑ (new): *accepting* / *corroborating* / *discounting*.
  - How far the nurse takes stated pain and severity at face value.
  - Corroborating is the ESI v5 norm: probe function, appearance and vitals.
  - Evidence: ENA 2023; Puntillo 2003; Marquié 2003; Capponi 2016; Grover 2012.
  - Observability: U (probing questions) plus E (decision given the reported pain).
- `questioning_style` (new; replaces most of `communication_style`): *open-facilitative* / *mixed* /
  *closed-directive*.
  - Open-facilitative: opens with a general inquiry, uses open questions, back-channels, summarises.
  - Closed-directive: confirmatory, yes/no and either/or questions.
  - Evidence: Heritage 2006a; Ernesäter 2014, 2016; Erkelens 2021; Lee 2015; Johri 2025.
  - Observability: U. Question form can be coded RIAS-style (Roter 2002).
  - Prevalence: closed questions dominate in telenursing (Ernesäter 2016).
- `emotional_responsiveness` (replaces `emotional_expression` and the affective part of
  `communication_style`): *task-focused* / *acknowledging* / *empathic*.
  - Levels are the VR-CoDES-P "reduces space", non-explicit and explicit "provides space" responses.
    Empathic responses also explain the process and the likely wait.
  - Evidence: Del Piccolo 2011; Ernesäter 2016; Street 2005; Mead 2000; Slade 2008; Janerka 2025.
  - Observability: U, and **only when the patient gave a cue** in the previous turn.
- `workload` ⚑ (episode context; replaces `verbosity`): *quiet* / *busy* / *overloaded*.
  - Shows up in interview length, pace, early closure and fewer empathic responses.
  - Evidence: ACEM 2023a, 2023b (2–5 minutes); Gerdtz 2001; Hitchcock 2014; Reay 2020; Fekonja 2023;
    Gorick 2026.
  - Observability: E (turns) plus U (explicit time pressure).
- `gender`: used only for the nurse's name and for TTS.
  - 88% of Australian nurses were female in 2023 (Department of Health n.d.).

## 5. Mapping from current fields

| Current field | Action | Proposed | Reason |
|---|---|---|---|
| P `age_group` | keep; take from case | `age_group` | Drives correlated sampling; the nurse sees it |
| P `gender` | keep | `gender` | Already matched to the case |
| P `ethnicity` | TTS only | — | Out of scope; already hidden |
| P `socioeconomic_status` | drop | → `health_literacy` | Not observable in a transcript; its effects run through literacy and trust (Street 2005; Armstrong 2007) |
| P `language_proficiency` | redefine | `english_proficiency` | Census and CEFR anchors |
| P `recall_accuracy` | redefine | `recall_reliability` (fabrication out of scope) | ED recall errors are omitted medicines, approximate details and changed answers (Mazer 2011; Caglar 2011; Iliceto 2016; Brandberg 2024) |
| P `cognitive_state` | redefine; case-conditioned | `cognitive_state` | CAM anchors; it is a triage sign (P1) |
| P `trust_in_healthcare` | redefine | `disclosure` (sensitive topics only) | ED under-reporting depends on the topic and on how the patient is asked (Rockett 2006; Rhodes 2007; Hankin 2015) |
| P `pain_expression` | split | `symptom_reporting` + `distress` | Reporting bias vs display (ESI/ATS pain rules) |
| P `reactivity_to_clinician_emotion` | merge | `distress` | Thin evidence; response to reassurance comes out of the interaction (Street 2005) |
| P `emotion_regulation` | merge | `distress` | State distress is the observable part |
| P `disfluency_rate` | keep | `disfluency` | Speech research; correlated with other traits |
| P `topic_drift` | rename | `focus` | Off-target verbosity (Arbuckle 1993) |
| P `verbosity` | keep | `verbosity` | Judged relative to question form |
| P — | new | `intoxication`, `demeanour`, `prior_ed_use` | ED prevalence (§2.1) |
| N `gender` | keep (name and TTS only) | `gender` | Not a behavioural trait |
| N `ethnicity` | TTS only | — | Out of scope |
| N `experience_level` | redefine | `expertise` | Changes style, not accuracy (Considine 2007) |
| N `risk_tolerance` | redefine | `risk_attitude` | Verified at episode level |
| N `guideline_adherence` | redefine | `algorithm_reliance` | Protocol vs intuition literature |
| N `communication_style` | split | `questioning_style` + `emotional_responsiveness` | CA, RIAS and VR-CoDES codes |
| N `verbosity` | merge | `workload` (+ `questioning_style`) | 2–5-minute norm; time pressure |
| N `emotional_expression` | merge | `emotional_responsiveness` | Coded response to cues |
| N — | new | `self_report_credence` | Counterpart to reported-pain rules |

## 6. Open questions

1. **Double counting pain.** MIMIC-IV-ED `pain` is already a patient self-report, shaped by the real
   patient's reporting style. "Faithful" should therefore mean reproducing the recorded score. Should
   amplification be applied on top of it?
2. **The judge conflicts with G traits.** Its "faithful" check (`JUDGE_CRITERIA["patient"]`) would
   reject intended minimising, amplifying, withholding or patchy recall. Its ground truth needs the
   intended transformation. E traits need aggregate checks instead:
   - over- and under-triage rate by `risk_attitude`;
   - `check_vital` use by `algorithm_reliance`;
   - question-type distribution by `questioning_style`.
3. **Third parties.** Interpreters and companions change triage talk (Flores 2005; Lee 2015). Should
   limited English proficiency or delirium add a third speaker, or stay out of scope?
4. **Weights without ED data.** `symptom_reporting`, `demeanour`, `verbosity`, `focus` and
   `disfluency` have no ED prevalence estimates. Use uniform weights (fine for control experiments),
   or expert elicitation for "realistic" corpora?
5. **Measuring bias.** The nurse sees age and gender. Documented under-triage of older, female and
   Black patients (Canto 2012; Grossmann 2012; Sax 2023; Arslanian-Engoren 2000) could serve as a
   bias probe for LLM nurses rather than something injected.
6. **ETEK version.** ETEK 2nd edition (ACSQHC 2024) supersedes the 2009 kit and ships pain-level and
   mental-health triage tools. Which edition are the cases seeded from?
7. **Traits that interact with the patient goal (⚑; flagged, not designed).**
   - The levers: `symptom_reporting` (the ESI ≥7/10 and ATS pain bands are the obvious levers),
     `distress` display, `disclosure` (strategic concealment), fabrication and `demeanour`
     (pressure).
   - The counter-levers: nurse `self_report_credence`, `risk_attitude`, `workload`, and
     `prior_ed_use` as reputation.
   - Evidence that the game exists in practice: nurses already suspect exaggeration (Capponi 2016),
     and classic "gaming" cues predict poorly (Grover 2012).
   - Separability: to identify strategic effects, the trait tendency (e.g. a catastrophizing
     amplifier) must be crossed with, not merged into, the goal.
   - Capability limits: patients who are delirious, markedly intoxicated, or have limited English or
     health literacy may lack the capacity or knowledge to play strategically, so goal-directed
     behaviour should be bounded for them.

## References

- ABS (2006). *Health Literacy, Australia, 2006* (cat. 4233.0). Australian Bureau of Statistics. https://www.abs.gov.au/ausstats/abs@.nsf/mf/4233.0
- ABS (2022). *Cultural diversity of Australia* (Census 2021). https://www.abs.gov.au/articles/cultural-diversity-australia
- ACEM (2023a). *Guidelines on the Implementation of the Australasian Triage Scale in Emergency Departments* (G24, V6). https://acem.org.au/getmedia/51dc74f7-9ff0-42ce-872a-0437f3db640a/G24_04_Guidelines_on_Implementation_of_ATS_Jul-16.aspx
- ACEM (2023b). *Policy on the Australasian Triage Scale* (P06, V5). https://policy.acem.org.au/index.php/policies-menu/p06-policy-on-the-australasian-triage-scale
- ACSQHC (2024). *Emergency Triage Education Kit (ETEK), second edition*. https://www.safetyandquality.gov.au/resources/emergency-triage-education-kit-etek-second-edition
- AIHW (2026). *Emergency department care* (2024–25 data). https://www.aihw.gov.au/hospitals/topics/emergency-departments
- Arbuckle TY, Gold DP (1993). Aging, inhibition, and verbosity. *J Gerontol* 48(5):P225–32. doi:10.1093/geronj/48.5.P225
- Armstrong K, Ravenell KL, McMurphy S, Putt M (2007). Racial/ethnic differences in physician distrust in the United States. *Am J Public Health* 97(7):1283–9. doi:10.2105/AJPH.2005.080762
- Arslanian-Engoren C (2000). Gender and age bias in triage decisions. *J Emerg Nurs* 26(2):117–24. doi:10.1016/S0099-1767(00)90053-9
- Barrows HS (1993). An overview of the uses of standardized patients for teaching and evaluating clinical skills. *Acad Med* 68(6):443–51. doi:10.1097/00001888-199306000-00002
- Barsky AJ, Goodson JD, Lane RS, Cleary PD (1988). The amplification of somatic symptoms. *Psychosom Med* 50(5):510–19. doi:10.1097/00006842-198809000-00007
- Barsky AJ (2002). Forgetting, fabricating, and telescoping: the instability of the medical history. *Arch Intern Med* 162(9):981–4. doi:10.1001/archinte.162.9.981
- Benner P (1982). From novice to expert. *Am J Nurs* 82(3):402–7. https://pubmed.ncbi.nlm.nih.gov/6917683/
- Bortfeld H, Leon SD, Bloom JE, Schober MF, Brennan SE (2001). Disfluency rates in conversation: effects of age, relationship, topic, role, and gender. *Lang Speech* 44(2):123–47. doi:10.1177/00238309010440020101
- Boudreaux ED, Camargo CA Jr, Arias SA, et al. (2016). Improving suicide risk screening and detection in the emergency department. *Am J Prev Med* 50(4):445–53. doi:10.1016/j.amepre.2015.09.029
- Brandberg H, Sundberg CJ, Spaak J, Koch S, Kahan T (2024). Are medical history data fit for risk stratification of patients with chest pain in emergency care? Comparing data collected from patients using computerized history taking with data documented by physicians in the electronic health record in the CLEOS-CPDS prospective cohort study. *J Am Med Inform Assoc* 31(7):1529–39. doi:10.1093/jamia/ocae110
- Caglar S, Henneman PL, Blank FS, Smithline HA, Henneman EA (2011). Emergency department medication lists are not accurate. *J Emerg Med* 40(6):613–6. doi:10.1016/j.jemermed.2008.02.060
- Canto JG, Rogers WJ, Goldberg RJ, et al. (2012). Association of age and sex with myocardial infarction symptom presentation and in-hospital mortality. *JAMA* 307(8):813–22. doi:10.1001/jama.2012.199
- Capponi R, Loguercio V, Guerrini S, et al. (2016). Does the Numeric Rating Scale (NRS) represent the optimal tool for evaluating pain in the triage process of patients presenting to the ED? *Acta Biomed* 87(3):347–52. https://pubmed.ncbi.nlm.nih.gov/28112706/
- Chen WJ, Fang CC, Shyu RS, Lin KC (2006). Underreporting of illicit drug use by patients at emergency departments as revealed by two-tiered urinalysis. *Addict Behav* 31(12):2304–8. doi:10.1016/j.addbeh.2006.02.015
- Cherpitel CJ, Ye Y, Bond J, et al. (2007). Validity of self-reported drinking before injury compared with a physiological measure: cross-national analysis of emergency-department data from 16 countries. *J Stud Alcohol Drugs* 68(2):296–302. doi:10.15288/jsad.2007.68.296
- Chung MK, Bartfield JM (2002). Knowledge of prescription medications among elderly emergency department patients. *Ann Emerg Med* 39(6):605–8. doi:10.1067/mem.2002.122853
- Cioffi J (1998). Decision making by emergency nurses in triage assessments. *Accid Emerg Nurs* 6(4):184–91. doi:10.1016/S0965-2302(98)90077-7
- Claassen CA, Larkin GL (2005). Occult suicidality in an emergency department population. *Br J Psychiatry* 186:352–3. doi:10.1192/bjp.186.4.352
- Considine J, LeVasseur SA, Villanueva E (2004). The Australasian Triage Scale: examining emergency department nurses' performance using computer and paper scenarios. *Ann Emerg Med* 44(5):516–23. doi:10.1016/j.annemergmed.2004.04.007
- Considine J, Botti M, Thomas S (2007). Do knowledge and experience have specific roles in triage decision-making? *Acad Emerg Med* 14(8):722–6. doi:10.1197/j.aem.2007.04.015
- Considine J, Oldland E, Currey J, et al. (2026). Emergency department triage nurses' scope of practice: an observational study. *J Clin Nurs* 35(9):3942–58. doi:10.1111/jocn.70368
- Cordell WH, Keene KK, Giles BK, et al. (2002). The high prevalence of pain in emergency medical care. *Am J Emerg Med* 20(3):165–9. doi:10.1053/ajem.2002.32643
- Craig KD (2009). The social communication model of pain. *Can Psychol* 50(1):22–32. doi:10.1037/a0014772
- Del Piccolo L, de Haes H, Heaven C, et al. (2011). Development of the Verona coding definitions of emotional sequences to code health providers' responses (VR-CoDES-P) to patient cues and concerns. *Patient Educ Couns* 82(2):149–55. doi:10.1016/j.pec.2010.02.024
- Department of Health (n.d.). *The value of nurses* (National Nursing Workforce Strategy). https://www.health.gov.au/resources/publications/national-nursing-workforce-strategyintroduction/the-value-of-nurses
- Egerton-Warburton D, Gosbell A, Moore K, et al. (2018). Alcohol-related harm in emergency departments: a prospective, multi-centre study. *Addiction* 113(4):623–32. doi:10.1111/add.14109
- ENA (2023). *Emergency Severity Index Handbook*, 5th ed. Emergency Nurses Association. https://californiaena.org/wp-content/uploads/2023/05/ESI-Handbook-5th-Edition-3-2023.pdf
- Erkelens DC, van Charldorp TC, Vinck VV, et al. (2021). Interactional implications of either/or-questions during telephone triage of callers with chest discomfort in out-of-hours primary care: a conversation analysis. *Patient Educ Couns* 104(2):308–14. doi:10.1016/j.pec.2020.07.011
- Ernesäter A, Engström M, Winblad U, Holmström IK (2014). A comparison of calls subjected to a malpractice claim versus 'normal calls' within the Swedish Healthcare Direct: a case–control study. *BMJ Open* 4(10):e005961. doi:10.1136/bmjopen-2014-005961
- Ernesäter A, Engström M, Winblad U, Rahmqvist M, Holmström IK (2016). Telephone nurses' communication and response to callers' concern—a mixed methods study. *Appl Nurs Res* 29:116–21. doi:10.1016/j.apnr.2015.04.012
- Fekonja Z, Kmetec S, Fekonja U, et al. (2023). Factors contributing to patient safety during triage process in the emergency department: a systematic review. *J Clin Nurs* 32(17–18):5461–77. doi:10.1111/jocn.16622
- Fleet RP, Dupuis G, Marchand A, et al. (1996). Panic disorder in emergency department chest pain patients: prevalence, comorbidity, suicidal ideation, and physician recognition. *Am J Med* 101(4):371–80. doi:10.1016/S0002-9343(96)00224-0
- Flores G (2005). The impact of medical interpreter services on the quality of health care: a systematic review. *Med Care Res Rev* 62(3):255–99. doi:10.1177/1077558705275416
- Fulbrook P, Lawrence P (2015). Survey of an Australian general emergency department: estimated prevalence of mental health disorders. *J Psychiatr Ment Health Nurs* 22(1):30–8. doi:10.1111/jpm.12191
- Geetha Manukumar A, Miller M, Patey C, et al. (2025). Privacy matters: experiences of rural and remote emergency department patients – a mixed-methods research conducted in Newfoundland and Labrador, Canada. *Health Serv Insights* 18:11786329251320431. doi:10.1177/11786329251320431
- Gerdtz MF, Bucknall TK (2001). Triage nurses' clinical decision making. An observational study of urgency assessment. *J Adv Nurs* 35(4):550–61. doi:10.1046/j.1365-2648.2001.01871.x
- Gerdtz MF, Collins M, Chu M, et al. (2008). Optimizing triage consistency in Australian emergency departments: the Emergency Triage Education Kit. *Emerg Med Australas* 20(3):250–9. doi:10.1111/j.1742-6723.2008.01089.x
- Göransson K, Ehrenberg A, Marklund B, Ehnfors M (2005). Accuracy and concordance of nurses in emergency department triage. *Scand J Caring Sci* 19(4):432–8. doi:10.1111/j.1471-6712.2005.00372.x
- Göransson KE, Ehnfors M, Fonteyn ME, Ehrenberg A (2008). Thinking strategies used by Registered Nurses during emergency department triage. *J Adv Nurs* 61(2):163–72. doi:10.1111/j.1365-2648.2007.04473.x
- Gorick H, McGee M, Smith TO (2026). Assessments under pressure: interviews with triage nurses in emergency departments. *J Adv Nurs* 82(6):6515–28. doi:10.1111/jan.70283
- Grossmann FF, Zumbrunn T, Frauchiger A, et al. (2012). At risk of undertriage? Testing the performance and accuracy of the Emergency Severity Index in older emergency department patients. *Ann Emerg Med* 60(3):317–25. doi:10.1016/j.annemergmed.2011.12.013
- Groves JE (1978). Taking care of the hateful patient. *N Engl J Med* 298(16):883–7. doi:10.1056/NEJM197804202981605
- Grover CA, Elder JW, Close RJ, Curry SM (2012). How frequently are "classic" drug-seeking behaviors used by drug-seeking patients in the emergency department? *West J Emerg Med* 13(5):416–21. doi:10.5811/westjem.2012.4.11600
- Hahn SR, Kroenke K, Spitzer RL, et al. (1996). The difficult patient: prevalence, psychopathology, and functional impairment. *J Gen Intern Med* 11(1):1–8. doi:10.1007/BF02603477
- Hall MA, Dugan E, Zheng B, Mishra AK (2001). Trust in physicians and medical institutions: what is it, can it be measured, and does it matter? *Milbank Q* 79(4):613–39. doi:10.1111/1468-0009.00223
- Han JH, Zimmerman EE, Cutler N, et al. (2009). Delirium in older emergency department patients: recognition, risk factors, and psychomotor subtypes. *Acad Emerg Med* 16(3):193–200. doi:10.1111/j.1553-2712.2008.00339.x
- Hankin A, Haley L, Baugher A, Colbert K, Houry D (2015). Kiosk versus in-person screening for alcohol and drug use in the emergency department: patient preferences and disclosure. *West J Emerg Med* 16(2):220–8. doi:10.5811/westjem.2015.1.24121
- Heritage J, Robinson JD (2006a). The structure of patients' presenting concerns: physicians' opening questions. *Health Commun* 19(2):89–102. doi:10.1207/s15327027hc1902_1
- Heritage J, Robinson JD (2006b). Accounting for the visit: giving reasons for seeking medical care. In Heritage J, Maynard DW (eds), *Communication in Medical Care*, Cambridge University Press. doi:10.1017/CBO9780511607172.005
- Herndon JB, Chaney M, Carden D (2011). Health literacy and emergency department outcomes: a systematic review. *Ann Emerg Med* 57(4):334–45. doi:10.1016/j.annemergmed.2010.08.035
- Hitchcock M, Gillespie B, Crilly J, Chaboyer W (2014). Triage: an investigation of the process and potential vulnerabilities. *J Adv Nurs* 70(7):1532–41. doi:10.1111/jan.12304
- Hustey FM, Meldon SW (2002). The prevalence and documentation of impaired mental status in elderly emergency department patients. *Ann Emerg Med* 39(3):248–53. doi:10.1067/mem.2002.122057
- Iliceto A, Berndt SL, Greenslade JH, et al. (2016). Agreement between patient-reported and cardiology-adjudicated medical history in patients with possible ischemic chest pain: an observational study. *Crit Pathw Cardiol* 15(3):121–5. doi:10.1097/HPC.0000000000000082
- Inouye SK, van Dyck CH, Alessi CA, et al. (1990). Clarifying confusion: the confusion assessment method. *Ann Intern Med* 113(12):941–8. doi:10.7326/0003-4819-113-12-941
- Janerka C, Leslie GD, Gill FJ, PCC ED Triage Group (2025). A cross-sectional survey reporting nurses' perspectives of person-centred care at emergency department triage and waiting room in Australia. *Australas Emerg Care* 28(4):280–6. doi:10.1016/j.auec.2025.05.004
- Johri S, Jeong J, Tran BA, et al. (2025). An evaluation framework for clinical use of large language models in patient interaction tasks. *Nat Med* 31(1):77–86. doi:10.1038/s41591-024-03328-5
- Karro J, Dent AW, Farish S (2005). Patient perceptions of privacy infringements in an emergency department. *Emerg Med Australas* 17(2):117–23. doi:10.1111/j.1742-6723.2005.00702.x
- Kimberg L, Vasquez JA, Sun J, et al. (2021). Fears of disclosure and misconceptions regarding domestic violence reporting amongst patients in two US emergency departments. *PLoS One* 16(12):e0260467. doi:10.1371/journal.pone.0260467
- Klein LR, Cole JB, Driver BE, et al. (2018). Unsuspected critical illness among emergency department patients presenting for acute alcohol intoxication. *Ann Emerg Med* 71(3):279–88. doi:10.1016/j.annemergmed.2017.07.021
- Kreshak AA, Wardi G, Tomaszewski CA (2015). The accuracy of emergency department medication history as determined by mass spectrometry analysis of urine: a pilot study. *J Emerg Med* 48(3):382–6. doi:10.1016/j.jemermed.2014.11.003
- Kyung D, Chung H, Bae S, et al. (2025). PatientSim: a persona-driven simulator for realistic doctor–patient interactions. NeurIPS 2025 Datasets & Benchmarks. arXiv:2505.17818. https://arxiv.org/abs/2505.17818
- LaCalle E, Rabin E (2010). Frequent users of emergency departments: the myths, the data, and the policy implications. *Ann Emerg Med* 56(1):42–8. doi:10.1016/j.annemergmed.2010.01.032
- Lee SH, Kim CW (2015). Presentation of patients' problems during triage in emergency medicine. *Patient Educ Couns* 98(5):578–87. doi:10.1016/j.pec.2015.01.011
- Levy AG, Scherer AM, Zikmund-Fisher BJ, et al. (2018). Prevalence of and factors associated with patient nondisclosure of medically relevant information to clinicians. *JAMA Netw Open* 1(7):e185293. doi:10.1001/jamanetworkopen.2018.5293
- Lewis KL, Bohnert CA, Gammon WL, et al. (2017). The Association of Standardized Patient Educators (ASPE) Standards of Best Practice (SOBP). *Adv Simul* 2:10. doi:10.1186/s41077-017-0043-4
- Limpawattana P, Phungoen P, Mitsungnern T, et al. (2016). Atypical presentations of older adults at the emergency department and associated factors. *Arch Gerontol Geriatr* 62:97–102. doi:10.1016/j.archger.2015.08.016
- Locker TE, Baston S, Mason SM, Nicholl J (2007). Defining frequent use of an urban emergency department. *Emerg Med J* 24(6):398–401. doi:10.1136/emj.2006.043844
- Luo X, Jiang X, Wu J (2026). MedDialBench: benchmarking LLM diagnostic robustness under parametric adversarial patient behaviors. arXiv:2604.06846. https://arxiv.org/abs/2604.06846
- Mahmoud I, Hou XY, Chu K, Clark M (2013). Language affects length of stay in emergency departments in Queensland public hospitals. *World J Emerg Med* 4(1):5–9. doi:10.5847/wjem.j.issn.1920-8642.2013.01.001
- Mansouri S, Marvania M, et al. (2026). VeriSim: a configurable framework for stress-testing medical AI under patient communication noise. arXiv:2604.10441. https://arxiv.org/abs/2604.10441
- Marquié L, Raufaste E, Lauque D, et al. (2003). Pain rating by patients and physicians: evidence of systematic pain miscalibration. *Pain* 102(3):289–96. doi:10.1016/S0304-3959(02)00402-5
- Mazer M, Deroos F, Hollander JE, et al. (2011). Medication history taking in emergency department triage is inaccurate and incomplete. *Acad Emerg Med* 18(1):102–4. doi:10.1111/j.1553-2712.2010.00959.x
- Mead N, Bower P (2000). Patient-centredness: a conceptual framework and review of the empirical literature. *Soc Sci Med* 51(7):1087–110. doi:10.1016/S0277-9536(00)00098-8
- MIT-LCP. MIMIC-IV `admissions` table documentation. https://mimic.mit.edu/docs/iv/modules/hosp/admissions.html
- Monte AA, Heard KJ, Hoppe JA, Vasiliou V, Gonzalez FJ (2015). The accuracy of self-reported drug ingestion histories in emergency department patients. *J Clin Pharmacol* 55(1):33–8. doi:10.1002/jcph.368
- Moser DK, Kimble LP, Alberts MJ, et al. (2006). Reducing delay in seeking treatment by patients with acute coronary syndrome and stroke: a scientific statement from the American Heart Association. *Circulation* 114(2):168–82. doi:10.1161/CIRCULATIONAHA.106.176040
- Murdoch J, Barnes R, Pooler J, et al. (2015). The impact of using computer decision-support software in primary care nurse-led telephone triage: interactional dilemmas and conversational consequences. *Soc Sci Med* 126:36–47. doi:10.1016/j.socscimed.2014.12.013
- Pearson SD, Goldman L, Orav EJ, et al. (1995). Triage decisions for emergency department patients with chest pain: do physicians' risk attitudes make the difference? *J Gen Intern Med* 10(10):557–64. doi:10.1007/BF02640365
- Platts-Mills TF, Travers D, Biese K, et al. (2010). Accuracy of the Emergency Severity Index triage instrument for identifying elder emergency department patients receiving an immediate life-saving intervention. *Acad Emerg Med* 17(3):238–43. doi:10.1111/j.1553-2712.2010.00670.x
- Puntillo K, Neighbor M, O'Neil N, Nixon R (2003). Accuracy of emergency nurses in assessment of patients' pain. *Pain Manag Nurs* 4(4):171–5. doi:10.1016/S1524-9042(03)00033-X
- Ramirez D, Engel KG, Tang TS (2008). Language interpreter utilization in the emergency department setting: a clinical review. *J Health Care Poor Underserved* 19(2):352–62. doi:10.1353/hpu.0.0019
- Reay G, Smith-MacDonald L, Then KL, Hall M, Rankin JA (2020). Triage emergency nurse decision-making: incidental findings from a focus group study. *Int Emerg Nurs* 48:100791. doi:10.1016/j.ienj.2019.100791
- Rhodes KV, Drum M, Anliker E, et al. (2006). Lowering the threshold for discussions of domestic violence: a randomized controlled trial of computer screening. *Arch Intern Med* 166(10):1107–14. doi:10.1001/archinte.166.10.1107
- Rhodes KV, Frankel RM, Levinthal N, et al. (2007). "You're not a victim of domestic violence, are you?" Provider–patient communication about domestic violence. *Ann Intern Med* 147(9):620–7. doi:10.7326/0003-4819-147-9-200711060-00006
- Rockett IRH, Putnam SL, Jia H, Smith GS (2006). Declared and undeclared substance use among emergency department patients: a population-based study. *Addiction* 101(5):706–12. doi:10.1111/j.1360-0443.2006.01397.x
- Roter D, Larson S (2002). The Roter interaction analysis system (RIAS): utility and flexibility for analysis of medical interactions. *Patient Educ Couns* 46(4):243–51. doi:10.1016/S0738-3991(02)00012-5
- Sax DR, Warton EM, Mark DG, et al. (2023). Evaluation of version 4 of the Emergency Severity Index in US emergency departments for the rate of mistriage. *JAMA Netw Open* 6(3):e233404. doi:10.1001/jamanetworkopen.2023.3404
- Schlager M, Jungmann F, Schmidgall S, et al. (2026). Patients With Personality: realistic patient simulation through controlled diversity and selective disclosure. arXiv:2606.17441. https://arxiv.org/abs/2606.17441
- Schmidgall S, Ziaei R, Harris C, et al. (2024). AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments. arXiv:2405.07960. https://arxiv.org/abs/2405.07960
- Slade D, Scheeres H, Manidis M, Iedema R, Dunston R (2008). Emergency communication: the discursive challenges facing emergency clinicians and patients in hospital emergency departments. *Discourse & Communication* 2(3):271–98. doi:10.1177/1750481308091910
- Steinhauser S, Haroz R, Jones I, et al. (2024). Emergency department staff compassion is associated with lower fear of enacted stigma among patients with opioid use disorder. *Acad Emerg Med* 31(12):1204–11. doi:10.1111/acem.14970
- Street RL Jr, Gordon HS, Ward MM, Krupat E, Kravitz RL (2005). Patient participation in medical consultations: why some patients are more involved than others. *Med Care* 43(10):960–9. doi:10.1097/01.mlr.0000178172.40344.70
- Sullivan MJL, Bishop SR, Pivik J (1995). The Pain Catastrophizing Scale: development and validation. *Psychol Assess* 7(4):524–32. doi:10.1037/1040-3590.7.4.524
- Sullivan MJL, Thorn B, Haythornthwaite JA, et al. (2001). Theoretical perspectives on the relation between catastrophizing and pain. *Clin J Pain* 17(1):52–64. doi:10.1097/00002508-200103000-00008
- Tam HL, Chung SF, Lou CK (2018). A review of triage accuracy and future direction. *BMC Emerg Med* 18(1):58. doi:10.1186/s12873-018-0215-0
- Todd KH, Ducharme J, Choiniere M, et al. (2007). Pain in the emergency department: results of the Pain and Emergency Medicine Initiative (PEMI) multicenter study. *J Pain* 8(6):460–6. doi:10.1016/j.jpain.2006.12.005
- Tu T, Schaekermann M, Palepu A, et al. (2025). Towards conversational diagnostic artificial intelligence. *Nature* 642(8067):442–50. doi:10.1038/s41586-025-08866-7
- Uscher-Pines L, Pines J, Kellermann A, et al. (2013). Emergency department visits for nonurgent conditions: systematic literature review. *Am J Manag Care* 19(1):47–59. https://pubmed.ncbi.nlm.nih.gov/23379744/
- Wang R, Milani S, Chiu JC, et al. (2024). PATIENT-Ψ: using large language models to simulate patients for training mental health professionals. *EMNLP 2024*, 12772–97. doi:10.18653/v1/2024.emnlp-main.711
- Yong HH (2006). Can attitudes of stoicism and cautiousness explain observed age-related variation in levels of self-rated pain, mood disturbance and functional interference in chronic pain patients? *Eur J Pain* 10(5):399–407. doi:10.1016/j.ejpain.2005.05.004
- Zachariasse JM, van der Hagen V, Seiger N, et al. (2019). Performance of triage systems in emergency care: a systematic review and meta-analysis. *BMJ Open* 9(5):e026471. doi:10.1136/bmjopen-2018-026471
- Zhu H, Shi X, Zhou J (2026). ELICITED: EHR-grounded longitudinal interactive conversations for information-seeking triage evaluation and decision-making (EHR2Dial-Triage). arXiv:2608.09024. https://arxiv.org/abs/2608.09024
- Zimmermann C, Del Piccolo L, Bensing J, et al. (2011). Coding patient emotional cues and concerns in medical consultations: the Verona coding definitions of emotional sequences (VR-CoDES). *Patient Educ Couns* 82(2):141–8. doi:10.1016/j.pec.2010.03.017
