# Software fixtures, not experimental findings

The three passages, three paired prompts, and six canned answers are original synthetic software-test material supplied under CC0. The numbered markers are synthetic references, not bibliography entries. Do not treat the demo as a TinyLlama run or use its proxy rates as research results.

The controlled example includes one entirely mismatched cited answer, one partially matched two-citation answer, an abstention cue without a citation, and an empty answer. Two answers are selected for correction. Correcting them changes one answer's any-match flag and two answers' all-candidate-match flags; the evaluation population stays at six.

`citation_cases.json` contains 24 labeled detector examples. These labels identify expected candidate formats, not whether cited sources are real. Passing these examples does not establish detector precision/recall in natural model outputs.
