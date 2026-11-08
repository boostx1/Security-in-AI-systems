

# Page 1

S¨akerhet i AI-system (DV2607)
Inl¨amningsuppgift
Martin Boldt
Sebastian Johansson
Sadi Alawadi
Anton Kanerva
Institutionen f¨or Datavetenskap (DIDA)
Blekinge Tekniska H¨ogskola
2026-09-21
Introduktion
I likhet med alla mjukvarubaserade system kan ¨aven maskininl¨arningsystem
(ML-system) och dess tr¨anade modeller attackeras av personer med illasinnat
upps˚at, t.ex. genom att skapa bakd¨orrar som lurar klassificeringsmodeller eller
som tar bort nyttan med ML-modellen. D¨arf¨or kr¨avs ocks˚a s¨akerhets˚atg¨arder
f¨or att kunna s¨akerst¨alla att ML-modellerna kan utf¨ora sina uppgifter p˚a de
s¨att de ¨ar ¨amnade att utf¨oras.
Denna inl¨amningsuppgift ¨ar uppdelad i tv˚a delar d¨ar ni i respektive del f˚ar stifta
bekantskap med tv˚a vanliga attackertyper p˚a ML-system, n¨amligen centralis-
erade Adversarial input attacker (i del 1) respektive distribuerade Adversarial
input attacker (i del 2). Notera att del 2 beskrivs p˚a engelska, medan del 1
beskrivs p˚a svenska.
Praktiskt uppl¨agg och betygs¨attning
Respektive del av inl¨amningsuppgiften inneh˚aller f¨orklaringar, uppgifter ni ska
utf¨ora samt fr˚agor ni ska besvara i en rapport. Rapporten betygss¨atts d¨arefer
med betygsskalan A-F. F¨or att erh˚alla n˚agot av betygen C, D eller E beh¨over ni
minst genomf¨ora de obligatoriska uppgifterna och fr˚agorna under Del 1 respek-
tive Del 2 nedanf¨or. F¨or att ha en chans att f˚a betyg A eller B kr¨avs dessutom
att ni genomf¨or det som beskrivs i de frivilliga delarna f¨or b˚ade Del 1 och Del
2 nedan.
Betygskriterierna ¨ar som f¨oljer. Om ni siktar p˚a betyg i skalan C, D eller E
m˚aste ni slutf¨ora uppgifterna 1.1 t.o.m. 1.3 och 2.1 t.o.m. 2.2. F¨or betyg B
m˚aste ni dessutom slutf¨ora uppgift 1.4 och 2.3. F¨or betyg A m˚aste ni dessutom
slutf¨ora uppgift 1.4 och 2.4.
Notera att det ¨ar obligatoriskt att jobba i grupper om tv˚a (2) studenter d˚a ni
genomf¨or denna inl¨amningsuppgift respektive skriver er gemensamma rapport.
N¨ar ni ¨ar klara med er skriftliga rapport l¨amnar ni in en ZIP-fil som inneh˚aller
er: (1) Pythonkod samt (2) er rapport i PDF-format. ZIP-filen skickas in p˚a
kursens Canvas-sida innan deadline, se mer info p˚a kurssidan. Innan ni skickar
in er inl¨amningsuppgift s˚a kontrollera att ni:
1. anv¨ander den obligatoriska rapportmallen som finns p˚a kurssidan,
1


# Page 2

2. utf¨orligt besvarat uppgifterna i b˚ade Del 1 och Del 2 nedan,
3. inkluderar all er Pythonkod som beh¨ovs f¨or att k¨ora era uppgifter (inkl.
Jupyter Notebooks fr˚an Del 2),
4. skrivit b˚ada personernas namn och e-postaddress i er rapport samt i head-
ern i all k¨allkod,
5. l¨amnar in er rapport i PDF-format, och
6. Packat ihop er kod och er PDF-rapport i en ZIP-fil som ni l¨amnar in.
2


# Page 3

Part 1: Adversarial input attacker och skydd
Evasion attack ¨ar en attack-typ d¨ar maskininl¨arningsmodellen attackeras genom
att modellen luras till att ta felaktiga beslut. N¨ar man talar om evasion attacks
syftar man oftast p˚a en attack som heter adversarial input attack. Adversarial
inputs ¨ar instanser vars v¨arden har manipulerats f¨or att klassificeras till en an-
nan klass utan att v¨ardena egentligen representerar den klassen. Attacken ¨ar
enklast utf¨ord och demonstrerad p˚a bilddata, d˚a man enkelt kan l¨agga ett s˚a
pass lite brus p˚a en bild att det ¨ar osynligt f¨or m¨annisko¨ogat, men f˚ar modellen
att klassificera bilden som n˚agot helt annat ¨an vad vi ser p˚a bilden. Den g˚ar
dock ¨aven att utf¨ora inom andra datatyper s˚asom text- eller tabelldata, ¨aven
om den senare medf¨or en del utmaningar f¨or att obfuskera ”bruset”.
Denna del av inl¨amningsuppgiften g˚ar ut p˚a att f˚a en grundl¨aggande f¨orst˚aelse
kring hur maskininl¨arningsmodeller kan attackeras via input i klassifikations-
steget genom adversarial input attacks, samt vilka s¨akerhets˚atg¨arder man kan
vidta f¨or att f¨orsv˚ara eller kanske t.o.m. f¨orhindra s˚adana attacker. Ni kom-
mer att tr¨ana en modell p˚a ett dataset f¨or att sedan attackera den resulterande
modellen med adversarial inputs som ni genererar f¨or att lura modellen. Ni ska
¨aven unders¨oka och l¨asa in er p˚a tillg¨angliga s¨akerhets˚atg¨arder mot adversarial
input attacks och redog¨ora f¨or dessa.
Uppgiften delas upp i f¨oljande tv˚a huvudsakliga delar: i) en teoretisk samman-
fattning om adversarial input attacks och s¨akerhets˚atg¨arder som kan/b¨or vid-
tas, samt ii) en praktisk implementation av attacken och en eller flera f¨orsvars-
mekanismer.
1.1 - Instudering
Innan ni implementerar attack och f¨orsvar f¨or adversarial input attacks ¨ar det
n¨odv¨andigt med en grundl¨aggande f¨orst˚aelse f¨or dessa. Ni ska h¨ar efterforska
lite mer i detalj hur adversarial input attacks g˚ar till och vad det finns f¨or skydd
mot dem. D˚a det finns flera olika varianter r¨acker det med att v¨alja en attacktyp
och en skydds˚atg¨ard. Skriv ditt svar i rapportens del 1.1.
1.2 - Implementation av attack
I detta steg ska ni implementera en attack som g¨or s˚a att bilden p˚a koalan
klassificeras som en traktor, utan att bilden ser annorlunda ut f¨or m¨anskliga
¨ogat. Mer info finns i Jupyter Notebooken under Part 1). Skriv er kod direkt
i Notebooken och exekvera den d¨ar s˚a att svaren syns i Notebooken n¨ar denna
l¨amnas in. S˚a l¨ange som ni kan f¨orklara metoden f˚ar ni anv¨anda valfri metod,
och s˚aklart ¨aven redan tillg¨angliga paket, t.ex. Adversarial Robustness Toolbox
(ART), f¨or att l¨osa uppgiften.
1.3 - S¨akerhets˚atg¨arder
I denna deluppgift ska ni v¨alja ett s˚a effektivt skydd som m¨ojligt mot just er
adversarial input attack. Eftersom det finns m˚anga olika typer av skydd ska ni
motivera varf¨or ni valt just det skydd ni gjort, ist¨allet f¨or n˚agot annat skydd.
Skriv ner era svar i rapportens del 1.3, och var noga med att motivera valet av
skyddd v¨al.
1.4 - Implementation av skydd (betyg A eller B)
I rapportens del 1.4 summerar ni vilket skydd ni uppn˚adde d˚a skyddsmekanis-
men hade implementerats. Det finns m˚anga olika varianter av skydd, och det ¨ar
vanligt att skydden resulterar i en ny modell som ¨ar h¨ardad mot attacker. Vid
implementation av ett s˚adant skydd redovisar ni en j¨amf¨orelse av resultaten
3


# Page 4

fr˚an er attack p˚a originalmodellen med resultaten p˚a den h¨ardade modellen.
Huvudsaken h¨ar ¨ar att ni tydligt kan uppvisa skyddets effekt och att ni kort
kan redog¨ora f¨or hur skyddsmekanismen fungerar.
Part 2: Attacks in federated learning scenario
In this task, you will use the FLOWER federated learning framework (https:
//flower.ai/docs/framework/index.html) and the CIFAR-10 dataset to sim-
ulate an attack scenario of your choice. The simulation will involve 35 commu-
nication rounds with 5 clients collaboratively training a global model. Run this
setup without any attack in order to have it as the baseline to compare with
later, and try to log the following metrics for each round:
1. Kappa
2. F1 Score
This baseline will be used for comparison for the rest of the assignment. Make
sure to use the FedAvg aggregation strategy and the IID partitioner that Flower
offers. You will have the opportunity to try out other aggregation and parti-
tioning setups later in the assignment.
Note: Flower generally encourages its users to use their CLI where the
simulations are run using superlink.
This is the supported approach
and is encouraged for those who want to learn the ”proper way” to use
Flower and who might want to use it in a future, more extensive project.
Those who want an approach that is easier to run in a notebook, or who
might struggle with getting superlink to run properly on their machine
can look to the deprecated ”flwr[simulation]”.
This tutorial provides
a good introduction on how to get started with Flower through their
CLI and this kaggle notebook provides a good reference on how to use
flwr[simulation].
Part 2.1: Mandatory task 1
Use the Flower framework to simulate a client-side attack of your choice and
see how it affects the global model performance across rounds. Run this attack
with both 1 and 2 attacked clients and compare with the baseline you recorded
in the last part.
In the report: Please explain how your attack works and briefly discuss how
it affected the performance of the global model.
Part 2.2: Mandatory task 2
In this part you are tasked with developing a defence against the attack you
proposed in the previous part.
Then explore the efficiency of the proposed
defence technique by running the code with 0, 1 and 2 attacked clients in order
to see how the defence affects the baseline and how well it deals with an increased
number of attacked clients.
In the report: Explain how your defence works. What are its strengths/weak-
nesses? Does the baseline performance degrade by using the defence? Does it
efficiently mitigate the effectiveness of the attack?
Part 2.3: Optional for A and/or B grade
For this task we will shift focus from defences and explore how different hy-
perparameters affect the robustness of our setup. You are to choose one of the
following aspects of the federated learning: aggregation strategy or data
4


# Page 5

partitioning. The baseline so far has been FedAvg and IID partitioning. You
are to explore 2 options for the one you choose. For each of these options you do
one run without any attack and one run with 1 or 2 attacked clients (how many
is up to you). Check out this link for a list of aggregation strategies offered
by Flower and this link for the data partitioning methods offered by Flower.
The ambitious student is of course welcome to design and explore their own
aggregation strategies or data partitioning methods.
In the report: Discuss how the alternatives you have chosen compare to the
baseline and to each other, both with respect to performance during normal
operation and with respect to the robustness of your global model when under
attack.
Some notes:
• You are welcome to explore other aspects of federated learning if
you wish. Just make sure to contact the person responsible for the
second part of the assignment and check first. Purely numerical
hyperparameters (like proximal µ in FedProx) will not be accepted.
• If you plan to explore aggregation strategies and your proposed
defence was a secure aggregation strategy, it will not count towards
the two aggregation strategies you are to explore for this task but
can make for a nice complement in analysis.
Part 2.4: Optional task for possibility to get grade A
Back to defences. Develop another defence technique against your chosen attack,
test it in the same settings as in part 2.2, and compare against the results
obtained by the previous defence technique. Please analyse the results and draw
a conclusion based on your own reflection. In the report: Explain your defence
and reason about its effectiveness. Was it better than your first one? Are there
any benefits/drawbacks compared to the one you developed previously?
Note: you still need to do part 2.3 for an option to get grade A.
In the end you will have to run the configurations in table 1. Additionally,
outside of the code and the report you are also to prepare a short presentation
(10-15 minutes) where you are to present the attack you choose, the defence
technique(s) and the results of the hyperparameter test if you did that part. Be
prepared to answer any questions that might appear after your presentation.
However, this presentation will be for the second part only.
Good luck!
5


# Page 6

Aggregation strategy
Partitioning
#attacked clients
Defence
Required for
FedAvg
IID
0
None
E-C
FedAvg
IID
1
None
E-C
FedAvg
IID
2
None
E-C
FedAvg
IID
0
Defence1
E-C
FedAvg
IID
1
Defence1
E-C
FedAvg
IID
2
Defence1
E-C
FedAvg/Alternative1
IID/Alternative1
0
None
B
FedAvg/Alternative1
IID/Alternative1
1 or 2
None
B
FedAvg/Alternative2
IID/Alternative2
0
None
B
FedAvg/Alternative2
IID/Alternative2
1 or 2
None
B
FedAvg
IID
0
Defence2
A
FedAvg
IID
1
Defence2
A
FedAvg
IID
2
Defence2
A
Table 1: The configurations you should run for part 2 of the assignment. The
first configuration is the baseline and will be used as a point of reference. The
focus parameter of each run is highlighted in bold.
6
