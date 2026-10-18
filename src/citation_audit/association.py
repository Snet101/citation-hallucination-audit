"""Conservative citation/claim association; never infer entailment from proximity."""

import re


def associate_candidates(text, candidates):
    code = [m.span() for m in re.finditer(r"```[\s\S]*?(?:```|$)|`[^`\n]*`", text)]
    candidates = [c for c in candidates if not any(a <= c["start"] < b for a, b in code)]
    lines = []
    offset = 0
    references = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip().strip("# ")
        if re.fullmatch(r"(?:references|sources|citing sources)\s*:?", stripped, re.I):
            references = True
        elif re.fullmatch(r"(?:examples?|explanation|answer|conclusion)\s*:?", stripped, re.I):
            references = False
        definition = re.match(r"\s*(?:[-*]\s*)?\[(\d+)\]\s+", line)
        lines.append((offset, offset + len(line), references, definition))
        offset += len(line)

    def sentence(candidate):
        start, end = candidate["start"], candidate["end"]
        previous = list(re.finditer(r"[.!?]\s+|\n", text[:start]))
        left = previous[-1].end() if previous else 0
        following = re.search(r"[.!?](?=\s|$)|\n", text[end:])
        right = end + following.end() if following else len(text)
        return {"start": left, "end": right, "text": text[left:right].strip()}

    details = []
    for candidate in candidates:
        line = next(item for item in lines if item[0] <= candidate["start"] < item[1])
        left, right, references, definition = line
        # A bibliography label is a definition, not a second citation occurrence.
        if definition and candidate["kind"] == "numbered":
            if candidate["start"] == left + definition.start(1) - 1:
                continue
        span = sentence(candidate)
        prose = span["text"]
        for other in candidates:
            prose = prose.replace(other["text"], " ")
        words = re.findall(r"[A-Za-z]+", prose)
        content = [
            w
            for w in words
            if w.lower()
            not in {"see", "source", "sources", "reference", "references", "documentation", "doi"}
        ]
        bibliography = references or definition is not None or len(content) < 2
        details.append((candidate, span, bibliography, definition.group(1) if definition else None))

    output = []
    for candidate, span, bibliography, label in details:
        if not bibliography:
            claims = [span]
            method = "inline_sentence"
        elif label is not None:
            # Only explicit, unambiguous single-number markers resolve definitions.
            claims = [
                body_span
                for body, body_span, is_reference, _ in details
                if not is_reference and body["kind"] == "numbered" and body["text"] == f"[{label}]"
            ]
            method = "explicit_numbered_reference" if claims else "unresolved"
        else:
            claims = []
            method = "unresolved"
        unique = {(c["start"], c["end"]): c for c in claims}
        claims = [unique[k] for k in sorted(unique)]
        output.append(
            {
                **candidate,
                "association_status": "resolved" if claims else "unresolved",
                "association_method": method,
                "claim_spans": claims,
                "claim_text": "\n".join(c["text"] for c in claims) if claims else None,
            }
        )
    return sorted(output, key=lambda c: c["start"])
