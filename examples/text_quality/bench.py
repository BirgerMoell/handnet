"""Score the hand-made good / middling / bad samples and print the detector breakdown."""
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", ".."))
from handnet.text_quality import HandNet

net = HandNet()
print("params", net.n_params, "nonzero", net.n_nonzero)

GOOD = {
 "swe": "Chur har varit bebodd sedan förromersk tid. Curia Raetorum var den romerska bosättningen. Den blev 451 biskopssäte. Den nuvarande domkyrkan färdigställdes 1272. Medan staden Chur blev protestantisk under reformationen förblev biskopshovet en katolsk enklav. Området utgjorde sedan 1514 ett rikslän direkt under Tysk-romerska riket och var därmed ett territorium som inte ingick i staden. Som exklav till Hof hörde huslängan Marsöl. Efter rikslänens avskaffande 1803 utgjorde Hof en egen kommun till 1852.",
 "eng": "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic organisms. Mitochondria generate most of the cell's supply of adenosine triphosphate, which is used as a source of chemical energy. They were first discovered by Albert von Kölliker in 1857 in the voluntary muscles of insects. The term mitochondrion was coined by Carl Benda in 1898, and the organelle has since become one of the most studied structures in cell biology.",
 "deu": "Die Stadt liegt am Zusammenfluss von Rhein und Mosel und ist seit der Römerzeit besiedelt. Im Mittelalter entwickelte sich Koblenz zu einem wichtigen Handelsplatz, und die Kurfürsten von Trier residierten hier über Jahrhunderte. Heute ist die Stadt vor allem für das Deutsche Eck bekannt, an dem ein Reiterstandbild Kaiser Wilhelms I. steht. Die Altstadt wurde nach dem Zweiten Weltkrieg weitgehend wiederaufgebaut.",
 "kat": "თბილისი საქართველოს დედაქალაქი და უდიდესი ქალაქია. იგი მდებარეობს მდინარე მტკვრის ორივე ნაპირზე. ქალაქი დაარსდა მეხუთე საუკუნეში მეფე ვახტანგ გორგასლის მიერ. თბილისი საუკუნეების განმავლობაში იყო მნიშვნელოვანი სავაჭრო ცენტრი აბრეშუმის გზაზე. დღეს ქალაქში ცხოვრობს მილიონზე მეტი ადამიანი, და იგი ქვეყნის პოლიტიკური, ეკონომიკური და კულტურული ცენტრია.",
 "fin": "Helsinki on Suomen pääkaupunki ja suurin kaupunki. Se sijaitsee Uudenmaan maakunnassa Suomenlahden rannalla. Kaupungin perusti Ruotsin kuningas Kustaa Vaasa vuonna 1550 kilpailemaan Tallinnan kanssa. Helsingistä tuli Suomen pääkaupunki vuonna 1812, kun Venäjän keisari Aleksanteri I siirsi pääkaupungin Turusta. Nykyään Helsingin seudulla asuu noin 1,5 miljoonaa ihmistä, ja se on maan merkittävin talous- ja kulttuurikeskus.",
 "jpn": "東京都は日本の首都であり、関東地方に位置する。人口は約千四百万人で、日本で最も人口の多い都道府県である。江戸時代には徳川幕府の本拠地として栄え、明治維新の後に東京と改称された。現在は政治、経済、文化の中心地として、世界有数の大都市のひとつに数えられている。市内には皇居や国会議事堂などの重要な施設が集まっている。",
 "zho": "北京是中华人民共和国的首都，也是全国的政治和文化中心。北京有三千多年的建城史和八百多年的建都史，是世界著名的古都。故宫、天坛、颐和园和长城等历史遗迹吸引了大量游客。今天的北京是一座现代化的国际大都市，拥有众多高等院校和科研机构。",
 "ara": "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها. تقع المدينة على ضفاف نهر النيل في شمال البلاد. تأسست القاهرة في القرن العاشر الميلادي على يد الفاطميين، وأصبحت منذ ذلك الحين مركزاً سياسياً وثقافياً مهماً في العالم العربي. تضم المدينة العديد من المعالم التاريخية مثل الأزهر الشريف وقلعة صلاح الدين.",
 "hin": "दिल्ली भारत की राजधानी है और देश के सबसे बड़े शहरों में से एक है। यह यमुना नदी के किनारे बसा हुआ है। दिल्ली का इतिहास बहुत पुराना है, और इसे कई बार बसाया और उजाड़ा गया है। आज यह शहर भारत का राजनीतिक, सांस्कृतिक और आर्थिक केंद्र है। यहाँ लाल किला, कुतुब मीनार और इंडिया गेट जैसे प्रसिद्ध स्मारक स्थित हैं।",
 "rus": "Санкт-Петербург был основан Петром Первым в 1703 году и более двухсот лет был столицей Российской империи. Город расположен в дельте Невы на берегу Финского залива. Его исторический центр внесён в список Всемирного наследия ЮНЕСКО. Сегодня Санкт-Петербург является вторым по численности населения городом России и крупным культурным центром.",
 "ell": "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας και μία από τις αρχαιότερες πόλεις του κόσμου. Η ιστορία της εκτείνεται σε περισσότερα από τρεις χιλιάδες χρόνια. Κατά την κλασική περίοδο, η Αθήνα υπήρξε το κέντρο της φιλοσοφίας, του θεάτρου και της δημοκρατίας. Σήμερα η πόλη αποτελεί το πολιτικό και οικονομικό κέντρο της χώρας.",
 "tha": "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย ตั้งอยู่ริมแม่น้ำเจ้าพระยาในภาคกลางของประเทศ เมืองนี้ก่อตั้งขึ้นในปี พ.ศ. 2325 โดยพระบาทสมเด็จพระพุทธยอดฟ้าจุฬาโลกมหาราช ปัจจุบันกรุงเทพมหานครเป็นศูนย์กลางทางการเมือง เศรษฐกิจ และวัฒนธรรมของประเทศ มีประชากรมากกว่าสิบล้านคน",
 "kor": "서울은 대한민국의 수도이며 가장 큰 도시이다. 한강을 중심으로 발달한 이 도시는 조선 시대부터 수도의 역할을 해왔다. 오늘날 서울은 정치, 경제, 문화의 중심지로서 세계적인 대도시 중 하나로 꼽힌다. 경복궁과 남산타워 같은 명소에는 매년 수많은 관광객이 찾아온다.",
 "tur": "İstanbul, Türkiye'nin en kalabalık şehri ve ekonomik, kültürel ve tarihi merkezidir. Şehir, Boğaziçi'nin iki yakasında, Avrupa ve Asya kıtalarında yer alır. Tarih boyunca Roma, Bizans ve Osmanlı imparatorluklarına başkentlik yapmıştır. Ayasofya, Topkapı Sarayı ve Sultanahmet Camii şehrin en önemli tarihi yapıları arasındadır.",
}

BAD = {
 "html": "<div class=\"nav\"><ul><li><a href=\"/home\">Home</a></li><li><a href=\"/about\">About</a></li></ul></div><script>var x = {a: 1, b: [2,3]};</script><p>&nbsp;</p>",
 "mojibake": "Vi Ã¤r glada att kunna meddela att fÃ¶retaget nu Ã¶ppnar ett nytt kontor i GÃ¶teborg. Det Ã¤r ett stort steg fÃ¶r oss, sÃ¤ger VD:n i ett pressmeddelande. â€Vi ser fram emot att vÃ¤xa.â€",
 "repeat": "wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow!!!!!!!!!!!!!!!!!!! aaaaaaaaaaaaaaaaaaaaaaaaaaa hahahahahahahahahahahahaha",
 "shouting": "BUY NOW!!! LIMITED OFFER!!! CLICK HERE FOR THE BEST DEALS ON CHEAP FLIGHTS HOTELS CARS!!! DON'T MISS OUT!!! CALL 0800-123-456 TODAY!!!",
 "table": "| id | name | value |\n|----|------|-------|\n| 1 | foo | 3.14 |\n| 2 | bar | 2.71 |\n| 3 | baz | 1.41 |\n| 4 | qux | 1.73 |\n| 5 | quux | 2.23 |",
 "listnav": "Home\nProducts\nServices\nAbout us\nContact\nLogin\nRegister\nTerms of service\nPrivacy policy\nCookie settings\nSitemap\nCareers\nPress\nInvestors\nSupport",
 "numbers": "2023-01-01 12:00:00 192.168.0.1 GET /index.html 200 1234\n2023-01-01 12:00:01 192.168.0.2 GET /about.html 404 0\n2023-01-01 12:00:02 192.168.0.3 POST /login 302 512",
 "keywords": "cheap flights, cheap hotels, cheap cars, flights hotels cars, best deals flights, deals hotels, deals cars, book now flights, book hotels, book cars, travel deals, travel flights, travel hotels",
 "urls": "Check http://example.com/a/b/c and https://foo.bar/baz?x=1&y=2 and http://spam.example/deal also https://another.link/here/now",
 "garbage": "��� text �� with lots of ��� replacement � characters �� everywhere ���",
 "code": "def foo(x):\n    return [i**2 for i in range(x) if i % 2 == 0]\n\nclass Bar(object):\n    def __init__(self, y):\n        self.y = y\n",
 "short": "Hello.",
 "mixed": "Detta är en text på svenska 这是一段中文 и немного по-русски και λίγα ελληνικά alles gemischt zusammen in einem Absatz som inte hänger ihop.",
 "nospace": "thisisaverylongstringwithoutanyspacesatallanditjustkeepsgoingandgoingandgoingwithnoendinsightwhichisnotgoodtext",
 "fragments": "Yes. No. Maybe. Ok. Go. Stop. Now. Here. There. Up. Down. Left. Right. In. Out. On. Off.",
 "wall": "det var en gång en liten pojke som bodde i en by vid skogen och han gick ut varje dag och lekte med sina vänner och de hade roligt och sedan gick de hem och åt middag och sedan sov de och nästa dag gjorde de samma sak igen och igen och igen utan att något hände och det fortsatte så hela sommaren och hösten",
 "emoji": "OMG 😂😂😂😂 this is sooo 🔥🔥🔥 best day ever 🎉🎉🎉🎉 love u all 💕💕💕💕💕 #blessed #yolo 😍😍😍",
}

MID = {
 "sms": "hej kan du köpa mjölk på vägen hem? vi har slut. kram",
 "forum": "I had the same problem last week. Turned out the driver was outdated, updating it fixed everything. Try that first before reinstalling.",
 "wiki_stub": "Löfsjön är en sjö i Gävle kommun i Gästrikland och ingår i Dalälvens huvudavrinningsområde. Sjön är 3 meter djup, har en yta på 0,24 kvadratkilometer och befinner sig 112 meter över havet.",
}

for name, grp in (("GOOD", GOOD), ("MID", MID), ("BAD", BAD)):
    print("==", name)
    for k, t in grp.items():
        ex = net.explain(t)
        s = ex.pop("score"); ex.pop("base")
        print("%-10s %5.1f  %s" % (k, s, " ".join("%s=%s" % kv for kv in ex.items())))

