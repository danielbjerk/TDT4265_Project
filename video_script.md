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
* Forventer mange store (biler) og små (mennesker) features, men få imellom. Kan 
* Biler er avlange i vannrett retning
* Mennesker er avlange i horisontal retning
* 