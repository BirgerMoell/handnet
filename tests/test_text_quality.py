import numpy as np

from handnet.text_quality import HandNet, encode, A

PROSE = {
    "swe": "Chur har varit bebodd sedan förromersk tid. Curia Raetorum var den romerska bosättningen. "
           "Den blev 451 biskopssäte. Den nuvarande domkyrkan färdigställdes 1272. Medan staden Chur blev "
           "protestantisk under reformationen förblev biskopshovet en katolsk enklav. Området utgjorde sedan "
           "1514 ett rikslän direkt under Tysk-romerska riket och var därmed ett territorium som inte ingick "
           "i staden. Efter rikslänens avskaffande 1803 utgjorde Hof en egen kommun till 1852.",
    "kat": "თბილისი საქართველოს დედაქალაქი და უდიდესი ქალაქია. იგი მდებარეობს მდინარე მტკვრის ორივე ნაპირზე. "
           "ქალაქი დაარსდა მეხუთე საუკუნეში მეფე ვახტანგ გორგასლის მიერ. თბილისი საუკუნეების განმავლობაში იყო "
           "მნიშვნელოვანი სავაჭრო ცენტრი აბრეშუმის გზაზე. დღეს ქალაქში ცხოვრობს მილიონზე მეტი ადამიანი, და იგი "
           "ქვეყნის პოლიტიკური, ეკონომიკური და კულტურული ცენტრია.",
    "jpn": "東京都は日本の首都であり、関東地方に位置する。人口は約千四百万人で、日本で最も人口の多い都道府県である。"
           "江戸時代には徳川幕府の本拠地として栄え、明治維新の後に東京と改称された。現在は政治、経済、文化の中心地として、"
           "世界有数の大都市のひとつに数えられている。市内には皇居や国会議事堂などの重要な施設が集まっている。",
    "ara": "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها. تقع المدينة على ضفاف نهر النيل في شمال البلاد. "
           "تأسست القاهرة في القرن العاشر الميلادي على يد الفاطميين، وأصبحت منذ ذلك الحين مركزاً سياسياً وثقافياً "
           "مهماً في العالم العربي. تضم المدينة العديد من المعالم التاريخية مثل الأزهر الشريف وقلعة صلاح الدين.",
    "hin": "दिल्ली भारत की राजधानी है और देश के सबसे बड़े शहरों में से एक है। यह यमुना नदी के किनारे बसा हुआ है। "
           "दिल्ली का इतिहास बहुत पुराना है, और इसे कई बार बसाया और उजाड़ा गया है। आज यह शहर भारत का राजनीतिक, "
           "सांस्कृतिक और आर्थिक केंद्र है। यहाँ लाल किला, कुतुब मीनार और इंडिया गेट जैसे प्रसिद्ध स्मारक स्थित हैं।",
}

JUNK = {
    "html": '<div class="nav"><ul><li><a href="/home">Home</a></li><li><a href="/about">About</a></li></ul>'
            '</div><script>var x = {a: 1, b: [2,3]};</script><p>&nbsp;</p>',
    "mojibake": "Vi Ã¤r glada att kunna meddela att fÃ¶retaget nu Ã¶ppnar ett nytt kontor i GÃ¶teborg. Det Ã¤r "
                "ett stort steg fÃ¶r oss, sÃ¤ger VD:n i ett pressmeddelande. â€Vi ser fram emot att vÃ¤xa.â€",
    "repeat": "wow wow wow wow wow wow wow wow wow wow!!!!!!!!!!!!!!!!!!! aaaaaaaaaaaaaaaaaaaaaaaaaaa hahahahahahaha",
    "table": "| id | name | value |\n|----|------|-------|\n| 1 | foo | 3.14 |\n| 2 | bar | 2.71 |\n| 3 | baz | 1.41 |",
    "log": "2023-01-01 12:00:00 192.168.0.1 GET /index.html 200 1234\n2023-01-01 12:00:01 192.168.0.2 GET /a 404 0",
    "garbage": "��� text �� with lots of ��� replacement � chars ��",
    "nospace": "thisisaverylongstringwithoutanyspacesatallanditjustkeepsgoingandgoingandgoingwithnoendinsight",
}


def test_shape_and_sparsity():
    net = HandNet()
    assert 95_000 <= net.n_params <= 105_000
    assert net.n_nonzero < 1000          # the logic lives in a few hundred weights
    assert encode("a b").shape == (3, A)


def test_prose_scores_high_in_every_script():
    net = HandNet()
    for lang, text in PROSE.items():
        assert net.score(text) >= 80, (lang, net.explain(text))


def test_junk_scores_low():
    net = HandNet()
    for kind, text in JUNK.items():
        assert net.score(text) < 50, (kind, net.explain(text))


def test_range_and_determinism():
    net = HandNet()
    for text in list(PROSE.values()) + list(JUNK.values()) + ["", "x", "\n\n\n"]:
        s = net.score(text)
        assert 0.0 <= s <= 100.0
        assert s == net.score(text)


def test_explain_adds_up():
    net = HandNet()
    for text in (PROSE["swe"], JUNK["html"]):
        ex = net.explain(text)
        total = ex["base"] + sum(v if k[0] == "+" else -v for k, v in ex.items() if k not in ("score", "base"))
        assert abs(min(max(total, 0), 100) - ex["score"]) < 0.5


def test_short_points_knob():
    sentence = "Hunden springer i parken och leker med bollen."
    assert HandNet(short_points=0).score(sentence) > HandNet().score(sentence) + 10


def test_rnn_closed_form_matches_loop():
    """The vectorised ReLU-RNN must equal the step-by-step recurrence."""
    net = HandNet()
    from handnet.text_quality import relu, K, D, C, H
    text = "Ett test. Två meningar!\nTredje raden utan punkt"
    x = encode(text) @ net.E
    T = x.shape[0]
    pad = np.zeros(((K - 1) // 2, D), dtype=np.float32)
    xp = np.concatenate([pad, x, pad])
    c = relu(sum(xp[k:k + T] @ net.Wc[k] for k in range(K)) + net.bc)
    h = np.zeros(H, dtype=np.float32)
    hs = []
    for t in range(T):
        h = relu(h @ net.Wh + c[t] @ net.Wx + net.bh)
        hs.append(h.copy())
    hs = np.stack(hs)
    f = net.features(text)
    assert np.allclose(f[C:C + H], hs.max(0), atol=1e-4)
    assert np.allclose(f[C + H:C + 2 * H], hs.mean(0), atol=1e-4)


def test_torch_export_matches_numpy():
    import pytest
    torch = pytest.importorskip("torch")
    net = HandNet()
    tnet = net.to_torch().eval()
    for text in (PROSE["swe"], JUNK["html"], PROSE["jpn"]):
        x = torch.tensor(encode(text))[None]
        with torch.no_grad():
            out = float(tnet(x)[0])
        assert abs(out - net.score(text)) < 1e-2, (text[:30], out, net.score(text))
