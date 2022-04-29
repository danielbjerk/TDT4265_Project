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

### Task 2.3
Her er det bare å presentere tensorboard utviklingen, og eventuelt forsøke å kjøre disse på et utvalg bilder. Forhåpentligvis sakte forbedring.


### Task 2.4
Momenter til spesialisering av modellen:
* Forventer mange store (biler) og små (mennesker) features, men få imellom. Er det mulig å lage dypere feature maps på endene men ikke i midten?
* Biler er avlange i vannrett retning
* Mennesker er avlange i horisontal retning, en fast aspect ratio på 0.46
* Mulig å sjekke distribusjonen av label størrelsene, lengde / høyde
* Kan vi endre alpha verdiene til å bedre representere vårt datasett? alpha er lagt til for å håndtere klasse ubalansen. Dersom vi skrur denne ned noe, kan vi også ende opp med bedre resultater, ettersom vi kan få mange falske positiver på de andre objekten også, ettersom det i mange av videoene ikke er så mange av de. Vi gir små vekter til de tingene som dukker opp ofte, og store vekter til de tingene som dukker opp lite. Her kan vi bruke grafene vi har funnet om hvilke objeter som dukker opp ofte og se om å distribuere disse alphaene slik at de går opp i 1 kan gi bedre resultater.
* Hva setter vi gamma til? I focal loss artikkelen sier de at de fikk best resultater med gamma = 2 men de sier at høyere verdier for gamma vil gjøre modellen bedre på vanskelige objekter, noe vi kanskje burde gjøre ettersom de vanskelige objektene har veldig liten nøyaktighet da de sjelden dukker opp i vårt datasett.

Før endringer: 
    AP for class background is -1.0000 (No objects of this class in validation set)
    AP for class car is 0.2405 
    AP for class truck is -1.0000 (No objects of this class in validation set)
    AP for class bus is 0.0000 
    AP for class motorcycle is -1.0000 (No objects of this class in validation set)
    AP for class bicycle is -1.0000 (No objects of this class in validation set)
    AP for class scooter is -1.0000 (No objects of this class in validation set)
    AP for class person is 0.0715 
    AP for class rider is 0.0001 
    2022-04-29 10:45:25,948 [INFO ] metrics/mAP: 0.078, metrics/mAP@0.5: 0.209, metrics/mAP@0.75: 0.033, metrics/mAP_small: 0.016, metrics/mAP_medium: 0.091, metrics/mAP_large: 0.227, metrics/average_recall@1: 0.025, metrics/average_recall@10: 0.102, metrics/average_recall@100: 0.139, metrics/average_recall@100_small: 0.065, metrics/average_recall@100_medium: 0.181, metrics/average_recall@100_large: 0.297, metrics/AP_car: 0.240, metrics/AP_bus: 0.000, metrics/AP_person: 0.072, metrics/AP_rider: 0.000, 
Etter endringer:


## Part 3

### Task 3.3
Svakhetene i denne tilnærmingen til dette prosjektet er at nøyaktighetene til prediksjonene er alt for lave til å kunne være i et faktisk produkt. Ettersom nøyaktigheten til bil ikke stiger så mye høyere enn 25% er det naturlig at hvor nyttig dette faktisk kan være for en kunde er diskutabelt. 
Weaknesses of the approach selected in this project is that the accuracy is way to low to actually make a difference. When the best accuracy is around 25% there a
    