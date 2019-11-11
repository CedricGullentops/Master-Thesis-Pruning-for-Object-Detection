# Presentatie

**Slide 1: **

Goedemorgen allemaal, ik ben Cedric Gullentops en welkom tot mijn tussentijdse presentatie over mijn masterproef: Het snoeien van convolutionele neurale netwerken voor objectdetectie op GPU.

---

**Slide 2: ** 

Hier ziet u een overzicht van wat ik zal bespreken. Eerst zal ik kort bespreken welke problemen zich voordoen bij het gebruik maken van convolutionele neurale netwerken en hoe mijn masterproef hier zich in situeert. Dan stel ik de principes van het snoeien van een netwerk voor en ten slotte stel ik mijn onderzoeksvraag.

Hierna zal ik aantonen hoe ik zal proberen deze problemen op te lossen door het verrichte onderzoek te bespreken en mijn manier van werken zal verklaren.

Ten slotte toon ik u de huidige planning.

---

**Slide 3: **

We zullen dus beginnen met de situering en de doelstelling.

---

**Slide 4: **

We zien tegenwoordig een sterke groei in het gebruik van machine learning. Voor het classificeren van afbeeldingen of video's worden er vooral neurale netwerken gebruikt, met name convolutionele neurale netwerken. Deze netwerken zijn er toe in staat om aan gezichtsherkenning te doen,  verschillende diersoorten te classificeren of objecten te herkennen. Deze netwerken bestaan uit verschillende lagen. 

Eerst en vooral worden convolutie lagen gebruikt voor het detecteren van kenmerken. De eerste convolutie lagen detecteren hierbij elementaire kenmerken zoals randen en lijnen. Convolutie lagen die verderop in het netwerk gelegen zijn detecteren complexere structuren.

Pooling lagen worden gebruikt om de dimensies van de invoer te verlagen. Max pooling zoals op deze afbeelding gebruikt telkens de maximale waarden uit een deelmatrix van de invoer.

Ten slotte worden volledig geconnecteerde lagen gebruikt om de invoer te classificeren. Hierbij wordt elke uitvoer van de andere lagen met elke neuroon geconnecteerd. Het resultaat wordt meestal nog eens herschaalt met behulp van een activatiefunctie. Zo wordt er voor elke klassen een kans voorspeld die zegt of de afbeelding deze klasse bevat of niet.  

De volledig geconnecteerde lagen gebruiken het meeste geheugen en hier wordt dus ook vaak gesnoeid maar het grootste deel van de berekeningen gebeurt in de convolutionele lagen. Het is dus ook het interessants en het moeilijkst om hier te snoeien. Onze focus zal dan liggen op het snoeien van deze lagen.

---

**Slide 5: **

Hier bespreek ik nog kort hoe zo'n convolutie werkt. Eerst wordt een convolutie filter verschoven over de invoer. Hierbij wordt elk element vermenigvuldit met de invoer en opgeteld resulterende in een kenmerkenmap. Als de invoer uit meerdere dimensies bestaat, zoals een kleurenafbeelding heeft 3 dimensies, dan zal de resulterende kenmerkenmap het resultaat van al deze dimensies bevatten. Er wordt een feature map aangemaakt per convolutiefilter in die laag.

---

**Slide 6: **

Door het veelzijdig gebruik van deze netwerken is er vraag naar implementatie op mobiele toestellen. Het probleem is hier echter dat deze toestellen beperkt zijn in geheugen, rekenkracht en energieopslag. Ook blijkt uit onderzoek dat deze netwerken veel redundantie bevatten. Dit wilt zeggen dat veel gewichten, filterwaarden en biassen geen grote bijdrage zullen leveren aan de classificatie van de invoer. Het is dus interessant om deze minder nuttige elementen te verwijderen uit het netwerk, wat we dus snoeien noemen. Na het snoeien zal het netwerk lichter zijn en algemener werken. Het is dus ook interessant om te snoeien als we te kampen krijgen met overfitting.

---

**Slide 7: **

We kunnen snoeien op 3 verschillende niveau's. Ten eerste kunnen we de gewichten of channels snoeien. Dit zijn de connecties tussen de neuronen hier aangegeven in het blauw. Ten tweede kunnen we de filters zelf gaan verwijderen. Bij verwijderde filters moeten we de connecties ook verwijderen zoals aangegeven staat in het rood. Ten slotte kunnen we intra-kernel of dus in de filter gaan snoeien. Dit doen we door de stride, of stappgrootte, aan te passen waardoor we een aantal convoluties zullen overslaan. Dit zorgt ervoor dat de resulterende kenmerkenmappen minder overeenkomen maar deze zullen dus ook kleiner worden.

---

**Slide 8: **

We kunnen het process van het snoeien van een netwerk als volgt voorstellen. Ten eerste moeten we de minst bijdragende componenten van het netwerk identificeren. Hierna moeten we deze verwijderen. Dit zal er ook wel voor zorgen dat er een vermindering in nauwkeurigheid optreedt. Ten slotte moeten we het netwerk hertrainen zodat de originele nauwkeurigheid terug bereikt kan worden.  
Het algemene doel van het snoeien kan met de volgende formule worden voorgesteld. Hierbij gaan we de nieuwe gewichten proberen te zoeken waarbij het verschil van de nauwkeurigheid van het nieuwe netwerk tegenover het vorige netwerk geminimaliseerd wordt. Tegelijkertijd proberen we het aantal niet-nul elementen in het nieuwe netwerk te verminderen.

Het snoeiprocess wordt hier nogmaals voorgesteld. Na het snoeien van het netwerk kunnen we ook nog gebruik maken van enkele andere technieken, om het geheugenverbruik te verminderen, zoals kwantisatie en Huffman encodering.

---

**Slide 9: **

Mijn onderzoeksvraag kan dan als volgt gesteld worden:

”Welke snoeitechnieken zijn het meest geschikt voor het snoeien van convolutionele neurale netwerken op GPU voor objectdetectie?”

Hierbij kunnen we nog twee deelvragen opstellen:

”Welke snoeitechnieken zijn het meest geschikt voor een vermindering in memory, en welke zijn het meest geschikt voor een vermindering in FLOP’s?”

”Tot in hoeverre kunnen we een CNN snoeien voordat er een vermindering van nauwkeurigheid optreed?”

---

**Slide 10: **

Ik heb nu de situering en doelstelling van mijn masterproef voorgesteld. Hierna zal ik bespreken hoe ik van plan ben hier een oplossing voor te bieden door de onderzochte literatuur te bespreken en mijn aanpak voor te stellen.

---

**Slide 11: **

Er was redelijk veel literatuur over het snoeien van een neuraal netwerk ter beschikking. Ik heb hier 3 technieken uitgekozen die ik elks zal bespreken.

Ten eerste heb ik gekozen voor de L2-norm. Herbij wordt er gesnoeid op basis van de grootte van de filtergewichten. Het idee hierbij is dat kleine filters zorgen voor kleine kenmerkenmappen en dus kleine activatiewaarden en vice versa. Hierbij zullen iteren over het netwerk en de kleinste filters verwijderen totdat het gewenste aantal filters verwijdert is. De grootte van een filter wordt bepaald door de L2-norm. De grootte van een filter is ook afhankelijk van de laag waarin hij zich bevindt. Het kan bijvoorbeeld zijn dat alle waarden in een bepaalde laag groot zijn. Alle filters moeten dus tegenover hun laag genormaliseerd worden om vergeleken te kunnen worden. Ook is er een verschil tussen hard en zacht snoeien. Bij hard snoeien zullen we de filters effectief verwijderen, bij zacht snoeien zetten we alle waarden van de filter op 0. Dit heeft als voordeel dat de filter hertraind kan worden maar het verbruikt vermogen wordt dan niet vrijgemaakt.

Deze formule toont aan hoe de L2-norm van een filter bereknt wordt. hierbij is w de gewichten van een filter en |w| de dimensionaliteit van de gewichten na vectorisatie.

---

**Slide 12: **

Hier ziet u enkele resultaten van het gebruik van deze techniek. Alle resultaten die ik vandaag nog toon zijn op de CIFAR-10 dataset gedaan. Wel met andere modellen/netwerken.

---

**Slide 13: **

Als tweede techniek maken we gebruik van de geometrische mediaan. Uit het onderzoek dat voorgesteld is door de autheurs van deze paper blijkt dat de distributies van de groottes van de filters vaak afwijkt van het ideale geval, wat dus een normale verdeling is. 

Op deze figuur kan u bijvoorbeeld zien dat de meeste filters zich in het midden bevinden terwijl slechts een klein aantal filters weinig bijdragen en weinig andere filters veel bijdragen. In de realiteit zien we dat sommige distributies dichter opeen liggen waardoor een afkappunt moeilijk er te bepalen is. In het andere geval is de kleinste norm van een filter nog steeds zeer groot. Ook hier is het dan moeilijk om te bepalen welke filters weg mogen.

---

**Slide 14: **

Deze techniek lost dit op door een punt x* te zoeken in de ruimte van filters in een laag, waarbij de Euclidische astand tussen deze filters geminimaliseerd wordt. Als de Euclidische afstand tussen dit punt en een filter klein is kunnen we stellen dat de informatie die deze filter bevat al aanwezig is in andere filters. Deze filter kan dan veilig verwijdert worden.

Dit wordt voorgesteld in de volgende formule. Hierbij zoeken we dus een punt x* waarbij het minimum van de functie wordt gezocht waarbij de functie de totale som is van de Euclidische afstanden van het punt x en de filters.

---

**Slide 15: **

Ook hier toon ik enkele resultaten uit het onderzoek met VGGNet op CIFAR-10.

---

**Slide 16: **

Als laatste hebben we de Centripetal SGD of Centripetal Stochastic Gradient Descent. Het idee bij deze techniek is om een aantal filters naar elkaar toe te laten groeien. Het aantal filters valt zelf te kiezen en de groepen worden bepaald met de k-means clustering methode. Als we deze filters naar elkaar laten groeien kunnen we de kenmerkenmappen van deze filters linken aan 1 filter en de andere filters verwijderen. We behouden met deze techniek dus de kenmerkenmappen van de filters.  Dit wordt op de volgende afbeelding opnieuw aangetoond.

---

**Slide 17: **

Het groeien van deze filters naar elkaar moet dus ook iteratief gebeuren. Hierbij maken we gebruik van een updateregel die de filters naar elkaar doet groeien. Hierbij F de kernel, Tau de groeisnelheid. L is de originele verliesfunctie. Nano is de gewichtsverliesfactor van het orignele model. E de centripetale kracht. 

De eerste term neemt het gemiddelde van de stijging van waarde door de verliesfunctie.  

De tweede term beschrijft de gewone gewichtsverlies functie.  

En ten slotte wordt het verschil van de originele waarden gradueel geeliminëerd.

---

**Slide 18: **

Ook hier zijn weer enkele resultaten van het onderzoek op het CIFAR-10 dataset. Zeer grote FLOP verlaging en soms zelfs met nauwkeurigheid verhoging!

---

**Slide 19: **

Het snoeien op basis van de L2-norm vormt een goede standaard voor de andere technieken mee te vergelijken. Het grootste nadeel van deze techniek is dat bij hard prunen het verwijderen van een verkeerde filter onomkeerbare effecten kan hebben op de nauwkeurigheid van het netwerk. Ook is er voor soft pruning nog bijkomende functionaliteiten nodig waar ik zo meteen nog verder op in ga.

De twee laatste technieken zijn state of the art technieken met zeer goede resultaten. Ze voorkomen de problemen van het snoeien op basis van een norm. Ze zijn echter wel complexer om te implementeren.

Het is ook zeer moeilijk om technieken te vergelijken met elkaar omdat deze vaak andere datasets of modellen gebruiken.

---

**Slide 20: **

Er waren natuurlijk ook nog andere technieken. Hier bespreek ik nog enkele andere met de reden waarom ik deze niet gekozen heb.

Ten eerste hebben we de Taylor expansie wat een methode is om de costfunctie van elke parameter in het netwerk in te schatten. Dit is een ook een uitstekende methode die veelgebruikt is maar het kamt met dezelfde problemen als het snoeien op basis van de norm, namelijk dat het snoeien van verkeerde filters onomkeerbare gevolgen kan hebben. Ook zijn er recente papers die gebruik maken van genetische algoritmes of meta-heuristische methodes zoals simulated annealing die geleidelijk aan naar een beter netwerk proberen te evolueren. Deze technieken zijn nog in een jonge fase en hadden behaalden niet dezelfde resultaten als de andere technieken. Ook wordt er onderzoek gedaan naar het gebruik van nieuwe soorten lagen, maar ook hier ga ik niet verder op in omdat de focus van het onderzoek ligt op convolutionele lagen.

---

**Slide 21: **

De implementatie van deze technieken worden op lightnet, wat op pytorch gebaseerd is, gemaakt omdat dit een goede basis vormt waarop netwerken aangemaakt en aangepast kunnen worden. Als netwerk hebben we gekozen voor het YOLOv2 netwerk. Hierbij is de nauwkeurigheid minder groot dan bij het YOLOv3 netwerk maar is de snelheid wel vele groter. En aangezien de focus weerom ligt op snelheid voor mobiele toestellen hebben we gekozen voor YOLOv2. De dataset waarop het netwerk getraind wordt is Pascal VOC. Als versiebeheer maak ik gebruik van git, met name github om aanpassingen aan het project te beheren. En de testen zullen worden uitgevoerd op een docker image om resultaten consistent te houden.

---

**Slide 22: **

Een eerst eimplementatie voor de L2-norm techniek is al reeds gemaakt.

Het gedeelte voor het hard snoeien werkt.

Bij het zacht snoeien was dus nog extra functionaliteit nodig. Als we namelijk de waarden van een bepaalde filter op 0 zetten wordt deze automatisch de kleinste. We moeten dus een manier hebben om op te slaan dat deze filter genegeerd moet worden. Dit wordt meestal gedaan een masker bij te houden en de structuur van de lagen aan te passen. Maar omdat dit niet gewenst is zullen we een andere methoden moeten vinden. Een optie zou zijn om een copy te nemen van het netwerk, dit hard te snoeien en bij te houden welke filters gesnoeid zijn, en nadien het copy zacht te snoeien.

---

**Slide 23: **

Nu heb ik u een overzicht gegeven van de te implementeren technieken en de gehandhaafde methoden. Nu zal ik nog kort even bespreken wat de planning betreft voor de resterende periode.

---

**Slide 24: **

We bevinden ons momenteel op half November. De basisfiles en omgeving van de implementatie zijn al gemaakt en de technieken zullen in de komende twee tot drie weken worden afgemaakt. Tijdens de implementatie van de laatste technieken zullen we al enkele testen uitvoeren en eventueel trachten enkele technieken te combineren. Ook kunnen we bijvoorbeeld eerst aan zacht snoeien doen en als laatse een keer hard snoeien. De implentatie en testen zouden afgerond moeten zijn tegen eind december zodat de laatste aanpassingen aan de masterproeftekst gemaakt kunnen worden. En dan geef ik u een eindpresentatie op 5 februari.

---

**Slide 25: **

Dan dank ik jullie voor jullie aandacht, zijn er nog vragen?

