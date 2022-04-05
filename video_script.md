Video script
# Video script

## Part 1 - Data exploration


* Ingen data med motorsykler så blir nærmest umulig å få trent opp nettverket 
* Flest forekomster av biler og personer, så er naturlig at det er disse som blir enklest å analysere og detektere når modellen skal testes. Det er ganske stor forskjell i størrelsesorden,.
* Personene dukker opp oftest lengst unna i bildet, lite detalj, og kan være store variasjoner i hva som er en person. 
* Forskjellen på en syklist og en person kan kanskje skape problemer
* Bildet blir tatt av en bil på veien, så vil jo personer og syklister ha et bias i størrelse siden de er lenger fra kameraet enn for eksempel biler. Så modellen vil tilpasse seg til bilder som er fra en bil på veien.
* Kan man kompansere ved å bruke sånne dærre RandomCrop, kompensere datasettet/størrelse på kjernen
* Ting som er nærme kameraet vil jo bli forvrengt, slik man ser når bilene passerer på venstre side. I et øyeblikk blir de forvrengt til en større størrelse enn de egentlig er.
* Ta med eksempelet hvor alle bysyklene ikke er med. 

## Part 2
### Task 2.4
Momenter til spesialisering av modellen:
* Forventer mange store (biler) og små (mennesker) features, men få imellom. Er det mulig å lage dypere feature maps på endene men ikke i midten?
* Biler er avlange i vannrett retning
* Mennesker er avlange i horisontal retning, en fast aspect ratio på 0.46
* Mulig å sjekke distribusjonen av label størrelsene, lengde / høyde
* Kan vi endre alpha verdiene til å bedre representere vårt datasett? alpha er lagt til for å håndtere klasse ubalansen. Dersom vi skrur denne ned noe, kan vi også ende opp med bedre resultater, ettersom vi kan få mange falske positiver på de andre objekten også, ettersom det i mange av videoene ikke er så mange av de. Vi gir små vekter til de tingene som dukker opp ofte, og store vekter til de tingene som dukker opp lite. Her kan vi bruke grafene vi har funnet om hvilke objeter som dukker opp ofte og se om å distribuere disse alphaene slik at de går opp i 1 kan gi bedre resultater.
* Hva setter vi gamma til? I focal loss artikkelen sier de at de fikk best resultater med gamma = 2 men de sier at høyere verdier for gamma vil gjøre modellen bedre på vanskelige objekter, noe vi kanskje burde gjøre ettersom de vanskelige objektene har veldig liten nøyaktighet da de sjelden dukker opp i vårt datasett.
