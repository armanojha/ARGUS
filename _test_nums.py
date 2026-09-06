from app.evaluation.answer_quality import _extract_numbers, _normalize_number, _numbers_match, _NUMBER_RE

text1 = "Revenue was " + chr(36) + "20M [1]."
text2 = "Revenue reached " + chr(36) + "20M in Q3."

print("text1:", repr(text1))
print("text2:", repr(text2))

m1 = _NUMBER_RE.findall(text1)
m2 = _NUMBER_RE.findall(text2)
print("regex matches t1:", m1)
print("regex matches t2:", m2)

n1 = _extract_numbers(text1)
n2 = _extract_numbers(text2)
print("extracted t1:", n1)
print("extracted t2:", n2)

if n1 and n2:
    print("normalize t1[0]:", _normalize_number(n1[0]))
    print("normalize t2[0]:", _normalize_number(n2[0]))
    print("match:", _numbers_match(n1[0], n2[0]))
