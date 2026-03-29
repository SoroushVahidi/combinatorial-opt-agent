# NLP4LP Manual Inspection Cases

These cases were selected by heuristic rules as likely hard for downstream
number-to-slot grounding. Each entry shows the problem text, numeric mentions,
and the reason the case was flagged.

---

## Entity-Association-Heavy Cases  (14 cases)

### 1. `nlp4lp_test_0`

**Why hard**: entity cues ['Mrs. Watson', 'Mrs. Watson'] with 5 numbers

**Text**: Mrs. Watson wants to invest in the real-estate market and has a total budget of at most $760000. She has two choices which include condos and detached houses. Each dollar invested in condos yields a $0.50 profit and each dollar invested in detached houses yields a $1 profit. A minimum of 20% of all money invested must be in condos, and at least $20

**Numeric mentions**: `$760000`, `$0.50`, `$1`, `20%`, `$20000`

---

### 2. `nlp4lp_test_6`

**Why hard**: entity cues ['Elm Furniture'] with 8 numbers

**Text**: A chair produced by Elm Furniture yields a profit of $43, while every dresser yields a $52 profit. Each week, 17 gallons of stain and 11 lengths of oak wood are available. Each chair requires 1.4 gallons of stain and 2 lengths of oak wood, while each dresser requires 1.1 gallons of stain and 3 lengths of oak wood. Determine the maximum profit.

**Numeric mentions**: `$43,`, `$52`, `17`, `11`, `1.4`, `2`, `1.1`, `3`

---

### 3. `nlp4lp_test_25`

**Why hard**: entity cues ['Oil Max', 'Oil Max Pro', 'Oil Max'] with 11 numbers

**Text**: A car manufacturer makes two types of car oils: Oil Max and Oil Max Pro. A container of Oil Max contains 46 grams of substance A, 43 grams of substance B and 56 grams of substance C. A container of Oil Max Pro contains 13 grams of substance A, 4 grams of substance B and 45 grams of substance C. The car manufacturer has 1345 grams of substance A, 34

**Numeric mentions**: `46`, `43`, `56`, `13`, `4`, `45`, `1345`, `346`, `1643`, `$10`

---

### 4. `nlp4lp_test_187`

**Why hard**: entity cues ['Election Day'] with 4 numbers

**Text**: A volunteer organization transports voters to the polls on Election Day either by vans or cars. They have vans which can carry 6 people and cars which can carry 3 people.  They need to transport at least 200 voters to the polls. In addition, at most 30% of the vehicles can be vans. How many of each vehicle should be used to minimize the total numbe

**Numeric mentions**: `6`, `3`, `200`, `30%`

---

### 5. `nlp4lp_test_196`

**Why hard**: entity cues ['Forest Paper'] with 7 numbers

**Text**: Forest Paper makes two types of products: graph paper and music paper. Each type of paper requires the use of two machines, a printing machine and a scanning machine. It takes 3 minutes on the printing machine and 5.5 minutes on the scanning machine to make a ream of graph paper. On the other hand, it takes 1.5 minutes on the printing machine and 3

**Numeric mentions**: `3`, `5.5`, `1.5`, `350`, `$4`, `$2.5`

---

### 6. `nlp4lp_test_199`

**Why hard**: entity cues ['Maple Oil'] with 11 numbers

**Text**: Maple Oil processes three types of crude oil: light oil, non-sticky oil and heavy oil. Each tank of light oil produces a net revenue of $550, each tank of non-sticky oil produces a net revenue of $750, and each tank of heavy oil produces a net revenue of $950. To process a tank of light oil, 3 units of compound A and 3 units of compound B are requi

**Numeric mentions**: `$550,`, `$750,`, `$950`, `3`, `6`, `2`, `9`, `250`, `150`

---

### 7. `nlp4lp_test_206`

**Why hard**: entity cues ['Super Shop'] with 8 numbers

**Text**: Super Shop sells cat paw snacks and gold shark snacks in bulk. It plans to sell them into two snack mix products. The first mix contains 20% cat paw snacks and 80% gold shark snacks. The second mix contains 35% cat paw snacks and 65% gold shark snacks. The store has on hand 20 kg of cat paw snacks and 50 kg of gold shark snacks. If the profit per k

**Numeric mentions**: `20%`, `80%`, `35%`, `65%`, `50`, `$12`, `$15,`

---

### 8. `nlp4lp_test_210`

**Why hard**: entity cues ['Platinum Database'] with 6 numbers

**Text**: Platinum Database sells two types of subscription software packages: a personal license and a commercial license which will cost $550 and $2000 to generate respectively. The marketing department estimates that they can sell at most 300 licenses for both versions combined a month. The profit per personal license is $450 and the profit per commercial

**Numeric mentions**: `$550`, `$2000`, `300`, `$450`, `$1200`, `$400000,`

---

### 9. `nlp4lp_test_211`

**Why hard**: entity cues ['Zeta Bakery'] with 10 numbers

**Text**: Zeta Bakery sells two types of cookies. They sell a strawberry cookie and a sugar cookie. Let's say they make x1 strawberry cookies, at a profit of $5.5 each, and x2 sugar cookies, at a profit of $12 each (x1 and x2 are unknowns both greater than or equal to 0). The daily demand for these cookies is at most 100 strawberry cookies and at most 80 sug

**Numeric mentions**: `1`, `$5.5`, `2`, `$12`, `0`, `100`, `80`

---

### 10. `nlp4lp_test_212`

**Why hard**: entity cues ['Beta Video', 'Gamma Live', 'Beta Video'] with 9 numbers

**Text**: A food company would like to run its commercials on three streaming platforms: Pi TV, Beta Video and Gamma Live. The cost for a commercial as well as the expected audience reach is given. On Pi TV, a commercial costs $1200 and attracts 2000 viewers. On Beta Video, a commercial costs $2000 and attracts 5000 viewers. On Gamma Live, a commercial costs

**Numeric mentions**: `$1200`, `2000`, `5000`, `$4000`, `9000`, `8`, `20%`, `$20000,`

---

### 11. `nlp4lp_test_307`

**Why hard**: entity cues ['This Manganese'] with 4 numbers

**Text**: PROBLEM TYPE: LP PROBLEM INFO:  - A foundry receives a specific order for \var{n_steel_quant} tons of steel.  - This steel must meet the following characteristics: it must contain at least \var{mn_percent} percentage of manganese (Mn) while its percentage of silicon (SI) must be between \var{si_min} and \var{si_max}. - To cast this steel, the found

**Numeric mentions**: `1,`

---

### 12. `nlp4lp_test_310`

**Why hard**: entity cues ['Wild Sports'] with 7 numbers

**Text**: PROBLEM TYPE: LP  PROBLEM INFO:  - Wild Sports produces \var{M} different products using \var{N} different raw materials. - The company has \var{available_{i}} of raw material \var{i} available. - Product \var{j} requires \var{req_{i,j}} units of material \var{i} per unit produced. - Product \var{j} has a selling price of \var{price_j} per unit. - 

**Numeric mentions**: `1,`

---

### 13. `nlp4lp_test_312`

**Why hard**: entity cues ['Custom Tees', 'Custom Tees'] with 15 numbers

**Text**: PROBLEM TYPE: LP  PROBLEM INFO:  - Custom Tees is planning an online advertising campaign with \var{A} different ad types across two web companies. - The company has set a goal of \var{goal_young} thousand clicks from visitors aged 18-25 and \var{goal_old} thousand clicks from visitors older than 25. - The company has set a goal of \var{goal_unique

**Numeric mentions**: `18`, `25`, `1000`, `1,`

---

### 14. `nlp4lp_test_318`

**Why hard**: entity cues ['Custom Tees', 'Custom Tees'] with 15 numbers

**Text**: PROBLEM TYPE: LP  PROBLEM INFO:  - Custom Tees is planning an online advertising campaign with \var{A} different ad types across two web companies. - The company has set a goal of \var{goal_young} thousand clicks from visitors aged 18-25 and \var{goal_old} thousand clicks from visitors older than 25. - The company has set a goal of \var{goal_unique

**Numeric mentions**: `18`, `25`, `1000`, `1,`

---

## Lower/Upper-Bound-Heavy Cases  (25 cases)

### 1. `nlp4lp_test_0`

**Why hard**: lower cues ['minimum', 'at least'] AND upper cues ['at most']

**Text**: Mrs. Watson wants to invest in the real-estate market and has a total budget of at most $760000. She has two choices which include condos and detached houses. Each dollar invested in condos yields a $0.50 profit and each dollar invested in detached houses yields a $1 profit. A minimum of 20% of all money invested must be in condos, and at least $20

**Numeric mentions**: `$760000`, `$0.50`, `$1`, `20%`, `$20000`

---

### 2. `nlp4lp_test_21`

**Why hard**: lower cues ['greater than or equal'] AND upper cues ['maximum']

**Text**: A car manufacturer makes two versions of the same car, a regular model and a premium model. They make x1 regular models per day and x2 premium models per day. The profit per regular model is $5000 and the profit per premium model is $8500 (x1 and x2 are unknown values both greater than or equal to 0). The daily demand for these cars is limited to a

**Numeric mentions**: `1`, `2`, `$5000`, `$8500`, `0`, `8`, `6`, `12`

---

### 3. `nlp4lp_test_29`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']

**Text**: Mark has 50 acres of land available to grow potatoes and cucumbers that he sells at a farmers' market. He must grow at least 12 acres of potatoes and 15 acres of cucumbers to meet his contract. Mark prefers to grow more cucumbers than potatoes, but he only has enough resources to grow at most twice the amount of cucumbers as potatoes. If the profit

**Numeric mentions**: `50`, `12`, `15`, `$500`, `$650,`

---

### 4. `nlp4lp_test_46`

**Why hard**: lower cues ['at least'] AND upper cues ['at most', 'must not exceed']

**Text**: An amusement park is installing ticket machines, both cash-based and card-only machines. A cash-based machine can process 20 people per hour while a card-only machine can process 30 people per hour. The cash-based machine needs 4 rolls of paper per hour while the card-only machine requires 5 rolls of paper per hour. The amusement park needs to be a

**Numeric mentions**: `20`, `30`, `4`, `5`, `500`, `90`

---

### 5. `nlp4lp_test_60`

**Why hard**: lower cues ['at least', 'at least'] AND upper cues ['at most']

**Text**: A laundromat can buy two types of washing machines, a top-loading model and a front-loading model. The top-loading model can wash 50 items per day while the front-loading model can wash 75 items per day. The top-loading model consumes 85 kWh per day while the front-loading model consumes 100 kWh per day. The laundromat must be able to wash at least

**Numeric mentions**: `50`, `75`, `85`, `100`, `5000`, `7000`, `40%`, `10`

---

### 6. `nlp4lp_test_72`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']

**Text**: An airport buys two types of vehicles, a 4-wheeler and 3-wheeler, to help move luggage. A 4-wheeler vehicle can move 60 luggage per day and produces 30 units of pollutant per day. A 3-wheeler vehicle can move 40 luggage per day and produces 15 units of pollutant per day. The airport needs to be able to move at least 1000 luggage per day. To avoid o

**Numeric mentions**: `4`, `3`, `60`, `30`, `40`, `15`, `1000`, `430`

---

### 7. `nlp4lp_test_86`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']

**Text**: An office is buying printers for their headquarters, a premium model and regular model. The premium model can print 30 pages per minute while the regular model can print 20 pages per minute. In addition, the premium model requires 4 units of ink per minute while the regular model requires 3 units of ink per minute. The office wants to make sure tha

**Numeric mentions**: `30`, `20`, `4`, `3`, `200`, `35`

---

### 8. `nlp4lp_test_94`

**Why hard**: lower cues ['at least', 'at least'] AND upper cues ['at most']

**Text**: A cleaning company uses a cleansing chemical and odor-removing chemical to clean a house. Each unit of the cleansing chemical takes 4 units to be effective while each unit of the odor-removing chemical takes 6 minutes to be effective. The company must use at least 100 units of the cleansing chemical. In total, at least 300 units of chemicals can be

**Numeric mentions**: `4`, `6`, `100`, `300`

---

### 9. `nlp4lp_test_104`

**Why hard**: lower cues ['at least'] AND upper cues ['at most', 'at most']

**Text**: Both fertilizer and seeds need to be added to a lawn. One unit of fertilizer takes 0.5 minutes to be effective while one unit of seeds takes 1.5 minutes to be effective. There can be at most 300 units of fertilizer and seeds combined added to the lawn. In addition at least 50 units of fertilizer need to be added. Since the lawn is really patchy, th

**Numeric mentions**: `0.5`, `1.5`, `300`, `50`

---

### 10. `nlp4lp_test_114`

**Why hard**: lower cues ['at least', 'at least'] AND upper cues ['at most']

**Text**: A patient takes anxiety medication and anti-depressants one after the other. Each unit of anxiety medication takes 3 minutes to be effective while each unit of anti-depressant takes 5 minutes to be effective. The patient must take at least 100 units of medication and at least 30 should be anxiety medication. Since the anxiety medication is strong, 

**Numeric mentions**: `3`, `5`, `100`, `30`

---

### 11. `nlp4lp_test_124`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']

**Text**: Both sulfate and ginger need to be added to a shampoo. One unit of sulfate takes 0.5 minutes to be effective while one unit of ginger takes 0.75 minutes to be effective. The shampoo must contain at least 100 units of sulfates and a total of 400 units of both ingredient. Since too much sulfate can damage the hair, there can be at most twice the amou

**Numeric mentions**: `0.5`, `0.75`, `100`, `400`

---

### 12. `nlp4lp_test_139`

**Why hard**: lower cues ['at least', 'at least'] AND upper cues ['at most', 'at most']

**Text**: A bee farmer transports his honey in small and large bottles. A small bottle can take 5 units of honey while a large bottle can take 20 units of honey. The farmer has available at most 300 small bottles and at most 100 large bottles. In addition, since small bottles are easier to sell, at least twice as many small bottles must be used than large bo

**Numeric mentions**: `5`, `20`, `300`, `100`, `200`, `50`

---

### 13. `nlp4lp_test_144`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']

**Text**: An aquarium does shows using otters and dolphins. An otter will do 3 tricks at a time and requires 3 treats to do so. A dolphin will do 1 trick at a time and requires 5 treats to do so. Since dolphins are more popular, at least 10 dolphins must be used and at most 30% of the performers can be otters. If the aquarium only has 200 treats available, m

**Numeric mentions**: `3`, `1`, `5`, `10`, `30%`, `200`

---

### 14. `nlp4lp_test_148`

**Why hard**: lower cues ['at least'] AND upper cues ['must not exceed']

**Text**: A meat shop ships their burger patties using refrigerated trucks and vans. Each truck can take 1000 patties at a cost of $300 per trip. Each van can take 500 patties at a cost of $100 per trip. Because the trucks have difficulty moving around in the city, the number of trucks must not exceed the number of vans. The meat shop has to ship at least 50

**Numeric mentions**: `1000`, `$300`, `500`, `$100`, `50000`, `$12500`

---

### 15. `nlp4lp_test_154`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']

**Text**: A meal service company delivers meals to customers either on electric bikes or scooters. A bike can hold 8 meals and requires 3 units of charge. A scooter can hold 5 meals and requires 2 units of charge. Since the city is more friendly towards scooters, at most 30% of the electric vehicles can be bikes and at least 20 scooters must be used. If the 

**Numeric mentions**: `8`, `3`, `5`, `2`, `30%`, `20`, `200`

---

### 16. `nlp4lp_test_160`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']

**Text**: A mail delivery service in an island village delivers mail by regular and speed boats. A regular boat can carry 20 pieces of mail per trip and uses 10 liters of gas. A speed boat can carry 30 pieces of mail per trip and uses 20 liters of gas. There can be at most 20 regular boat trips. Since customers want their mail as fast as possible, at least 5

**Numeric mentions**: `20`, `10`, `30`, `50%`, `1000`

---

### 17. `nlp4lp_test_165`

**Why hard**: lower cues ['at least', 'minimum'] AND upper cues ['cannot exceed']

**Text**: A jam company sends its product out in small and large jars. A small jar can hold 50 ml of jam while a large jar can hold 200 ml of jam. Most store prefer the smaller size and so the number of large jars cannot exceed the number of small jars. If the company wants to ship at least 100000 ml of jam, find the minimum number of jars that can be used.

**Numeric mentions**: `50`, `200`, `100000`

---

### 18. `nlp4lp_test_170`

**Why hard**: lower cues ['minimum', 'at least'] AND upper cues ['at most']

**Text**: A tropical city full of islands sends mail either by submarine or by boat. A submarine can carry 100 pieces of mail per trip and uses 30 liters of gas. A boat can carry 80 pieces of mail per trip and uses 25 liters of gas. There can be at most 6 submarine trips and a minimum of 50% of the trips must be by boat. If the city needs to transport at lea

**Numeric mentions**: `100`, `30`, `80`, `25`, `6`, `50%`, `1000`

---

### 19. `nlp4lp_test_175`

**Why hard**: lower cues ['at least', 'minimum'] AND upper cues ['cannot exceed', 'at most']

**Text**: A construction company in the tropics uses cows and elephants to carry bricks. A cow can carry 20 bricks on its back while an elephant can carry 50 bricks on its back. To avoid having elephants create too much traffic, the number of elephant cannot exceed the number of cows. In addition, there can be at most twice the number of cows as elephants. I

**Numeric mentions**: `20`, `50`, `1000`

---

### 20. `nlp4lp_test_181`

**Why hard**: lower cues ['At least', 'at least'] AND upper cues ['at most']

**Text**: There has been a horrible accident and patients need to be taken to the hospital by either a helicopter or bus. A helicopter can transport 5 patients per trip and takes 1 hour. On the other hand, a bus can transport 8 patients per trip and takes 3 hours. At least 120 patients need to be transported and at least 30% of the trips should be by helicop

**Numeric mentions**: `5`, `1`, `8`, `3`, `120`, `30%`, `10`

---

### 21. `nlp4lp_test_187`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']

**Text**: A volunteer organization transports voters to the polls on Election Day either by vans or cars. They have vans which can carry 6 people and cars which can carry 3 people.  They need to transport at least 200 voters to the polls. In addition, at most 30% of the vehicles can be vans. How many of each vehicle should be used to minimize the total numbe

**Numeric mentions**: `6`, `3`, `200`, `30%`

---

### 22. `nlp4lp_test_208`

**Why hard**: lower cues ['at least', 'at least'] AND upper cues ['at most', 'at most']

**Text**: A printing company sells math workbooks and English workbooks. To meet demand, they must make at least 40 math workbooks and at least 60 English workbooks. However, they can make at most 140 math workbooks and at most 170 English workbooks. The company has a contract with a school to send at least 200 workbooks of either type. If the profit per mat

**Numeric mentions**: `40`, `60`, `140`, `170`, `200`, `$15`, `$17,`

---

### 23. `nlp4lp_test_215`

**Why hard**: lower cues ['min', 'min'] AND upper cues ['maximum']

**Text**: A concert organizer has to transport equipment using carts or trolleys. Carts can transport 5 kg/min of equipment and requires 2 workers. Trolleys can transport 7 kg/min of equipment and requires 4 workers. There must be at least 12 trolleys to be used. Additionally, only a maximum of 40% of the transportation can be using trolleys. The organizer h

**Numeric mentions**: `5`, `2`, `7`, `4`, `12`, `40%`, `100`

---

### 24. `nlp4lp_test_222`

**Why hard**: lower cues ['at least', 'minimum'] AND upper cues ['at most']

**Text**: An oil and gas company has two types of pipes, a high-volume and a low-volume one. Every day, the high-volume pipe allows 10000 US gallons and it is recommended that 12 technicians closely monitor the pipes to ensure that it is functioning properly. Each day, the low-volume pipe allows 5000 US gallons and 5 technicians should closely monitor for sa

**Numeric mentions**: `10000`, `12`, `5000`, `5`, `150000`, `160`, `35`, `8`

---

### 25. `nlp4lp_test_235`

**Why hard**: lower cues ['at least', 'at least'] AND upper cues ['at most']

**Text**: A lighting company has access to two types of lights to provide their customers, an LED fixture, and a fluorescence lamp. The LED light uses 5 units of electricity per hour and needs to be changed 3 times a decade. Conversely, the fluorescence lamp uses 8 units of electricity per hour and needs to be changed 4 times a decade. Due to previous instal

**Numeric mentions**: `5`, `3`, `8`, `4`, `30%`, `300`, `2000`

---

## Multi-Number Confusion Cases  (25 cases)

### 1. `nlp4lp_test_0`

**Why hard**: 5 distinct numbers: ['$760000', '$0.50', '$1', '20%', '$20000']

**Text**: Mrs. Watson wants to invest in the real-estate market and has a total budget of at most $760000. She has two choices which include condos and detached houses. Each dollar invested in condos yields a $0.50 profit and each dollar invested in detached houses yields a $1 profit. A minimum of 20% of all money invested must be in condos, and at least $20

**Numeric mentions**: `$760000`, `$0.50`, `$1`, `20%`, `$20000`

---

### 2. `nlp4lp_test_11`

**Why hard**: 5 distinct numbers: ['300', '3', '5', '8', '$10']

**Text**: A glass factory makes two types of glass panes: a regular glass pane and a tempered glass pane. Both require time on a heating and cooling machine. Both machines are available for a maximum of 300 minutes per day. It takes 3 minutes in the heating machine and 5 minutes in the cooling machine to make one regular glass pane. It takes 5 minutes in the

**Numeric mentions**: `300`, `3`, `5`, `8`, `$10`

---

### 3. `nlp4lp_test_22`

**Why hard**: 8 distinct numbers: ['$2000', '$300,', '10', '6', '$100,', '4']

**Text**: You are designing an office space with two types of desks: long desks and short desks. You can spend at most $2000. Long desks cost $300, take up 10 square feet of space, and seat 6 employees. Short desks cost $100, take up 4 square feet of space, and seat 2 employees. The office can have at most 200 square feet of desks. How many of each desk shou

**Numeric mentions**: `$2000`, `$300,`, `10`, `6`, `$100,`, `4`, `2`, `200`

---

### 4. `nlp4lp_test_33`

**Why hard**: 8 distinct numbers: ['$200', '$300', '2', '3', '4', '5']

**Text**: A company sells custom scooters and bikes for customers. The profit per scooter is $200 and the profit per bike is $300. Each product requires time with the design team and engineering team. Each scooter needs 2 hours with the design team and 3 hours with the engineering team. Each bike needs 4 hours with the design team and 5 hours with the engine

**Numeric mentions**: `$200`, `$300`, `2`, `3`, `4`, `5`, `5000`, `6000`

---

### 5. `nlp4lp_test_44`

**Why hard**: 9 distinct numbers: ['200', '10', '8', '5', '2', '3']

**Text**: A farmer has 200 acres of land on which he must process hay using either a windrower or hay harvester. For each acre of land, the windrower can process 10 kg of hay while the hay harvester can process 8 kg of hay. Per acre, the windrower produces 5 kg of methane gas and requires 2 kg of fuel. On the other hand, the hay harvester produces 3 kg of me

**Numeric mentions**: `200`, `10`, `8`, `5`, `2`, `3`, `1`, `300`, `800`

---

### 6. `nlp4lp_test_55`

**Why hard**: 8 distinct numbers: ['5', '8', '3', '6', '600', '800']

**Text**: A bakery makes almond and pistachio croissants. An almond croissant requires 5 units of butter and 8 units of flour. A pistachio croissant requires 3 units of butter and 6 units of flour. The bakery has available 600 units of butter and 800 units of flour. Since the almond croissant is more popular, at least 3 times as many almond croissants should

**Numeric mentions**: `5`, `8`, `3`, `6`, `600`, `800`, `12`, `10`

---

### 7. `nlp4lp_test_66`

**Why hard**: 6 distinct numbers: ['50', '30', '20', '15', '300', '135']

**Text**: A post office is buying stamping machines and they can buy a dual or single model stamping machine. A dual model stamping machine can stamp 50 letters per minute while a single model stamping machine can stamp 30 letters per minute. The dual model stamping machine requires 20 units of glue per minute while the single model stamping machine requires

**Numeric mentions**: `50`, `30`, `20`, `15`, `300`, `135`

---

### 8. `nlp4lp_test_77`

**Why hard**: 5 distinct numbers: ['500', '750', '3', '20', '250000']

**Text**: A water company sells water in glass and plastic bottles. A glass bottle can hole 500 ml of water while a plastic bottle can hold 750 ml of water. Because most customer prefer plastic bottles, the number of plastic bottles must be at least 3 times the number of glass bottles. However, there must be at least 20 glass bottles. If the company has avai

**Numeric mentions**: `500`, `750`, `3`, `20`, `250000`

---

### 9. `nlp4lp_test_88`

**Why hard**: 6 distinct numbers: ['3', '5', '4', '3000', '4000', '30%']

**Text**: A candy company is making peach flavored candy and cherry flavored candy. Each pack of peach flavored candy requires 3 units of peach flavoring and 5 units of special syrup. Each pack of cherry flavored candy requires 5 units of cherry flavoring and 4 units of special syrup. The company has available 3000 units of peach flavoring and 4000 units of 

**Numeric mentions**: `3`, `5`, `4`, `3000`, `4000`, `30%`

---

### 10. `nlp4lp_test_99`

**Why hard**: 8 distinct numbers: ['1', '2', '0.5', '0.4', '0.2', '0.3']

**Text**: A patient with a sore throat can drink two syrups, syrup 1 and syrup 2 for treatment. Per serving, syrup 1 delivers 0.5 units of medicine to the throat and 0.4 units of medicine to the lungs. Per serving, syrup 2 delivers 0.2 units of medicine to the throat and 0.5 units of medicine to the lungs. Furthermore, syrup 1 contains 0.5 units of sugar whi

**Numeric mentions**: `1`, `2`, `0.5`, `0.4`, `0.2`, `0.3`, `5`, `4`

---

### 11. `nlp4lp_test_110`

**Why hard**: 9 distinct numbers: ['400', '20', '100', '300', '10', '75']

**Text**: A travelling salesman only eats ramen and fries. Each pack of ramen contains 400 calories, 20 grams of protein, and 100 mg of sodium. Each pack of fries contains 300 calories, 10 grams of protein, and 75 mg of sodium. Since fries are easier to eat while driving, at most 30% of his meals can be ramen. The salesman wants to ensure he eats at least 30

**Numeric mentions**: `400`, `20`, `100`, `300`, `10`, `75`, `30%`, `3000`, `80`

---

### 12. `nlp4lp_test_121`

**Why hard**: 10 distinct numbers: ['1,', '3', '5', '4', '2,', '8']

**Text**: In a science club, there are two tables that can be set up to make slime. At table 1, 3 units of powder and 5 units of glue are used to make 4 units of slime. At table 2, 8 units of powder and 6 units of glue are used to make 5 units of slime. However, table 1 produces 2 units of mess while table 2 produces 4 units of mess. The science club has ava

**Numeric mentions**: `1,`, `3`, `5`, `4`, `2,`, `8`, `6`, `100`, `90`, `30`

---

### 13. `nlp4lp_test_132`

**Why hard**: 6 distinct numbers: ['1000', '3', '2', '1', '100', '60%']

**Text**: A lab has 1000 units of medicinal ingredients to make two pills, a large pill and a small pill. A large pill requires 3 units of medicinal ingredients and 2 units of filler. A small pill requires 2 units of medicinal ingredients and 1 unit of filler. The lab has to make at least 100 large pills. However, since small pills are more popular at least 

**Numeric mentions**: `1000`, `3`, `2`, `1`, `100`, `60%`

---

### 14. `nlp4lp_test_143`

**Why hard**: 7 distinct numbers: ['3', '50', '5', '80', '150', '6']

**Text**: A lawn mowing service provides neighborhood services using small teams and large teams. A small team requires 3 employees and can mow 50 sq ft of lawn. A large team requires 5 employees and can mow 80 sq ft of lawn. The company has 150 employees available. Because most people have smaller lawns in the city, the number of small teams must be at leas

**Numeric mentions**: `3`, `50`, `5`, `80`, `150`, `6`, `10`

---

### 15. `nlp4lp_test_154`

**Why hard**: 7 distinct numbers: ['8', '3', '5', '2', '30%', '20']

**Text**: A meal service company delivers meals to customers either on electric bikes or scooters. A bike can hold 8 meals and requires 3 units of charge. A scooter can hold 5 meals and requires 2 units of charge. Since the city is more friendly towards scooters, at most 30% of the electric vehicles can be bikes and at least 20 scooters must be used. If the 

**Numeric mentions**: `8`, `3`, `5`, `2`, `30%`, `20`, `200`

---

### 16. `nlp4lp_test_165`

**Why hard**: 3 distinct numbers: ['50', '200', '100000']

**Text**: A jam company sends its product out in small and large jars. A small jar can hold 50 ml of jam while a large jar can hold 200 ml of jam. Most store prefer the smaller size and so the number of large jars cannot exceed the number of small jars. If the company wants to ship at least 100000 ml of jam, find the minimum number of jars that can be used.

**Numeric mentions**: `50`, `200`, `100000`

---

### 17. `nlp4lp_test_176`

**Why hard**: 4 distinct numbers: ['20', '15', '4', '500']

**Text**: A farmer on an island sends corn to the main land either by ferry or light rail. Each ferry trip can take 20 boxes of corn while each light rail trip can take 15 boxes of corn. Since ferry trips are slow, the number of light rail trip has to be at least 4 times the number of ferry trips. If the farmer wants to send at least 500 boxes of corn, minim

**Numeric mentions**: `20`, `15`, `4`, `500`

---

### 18. `nlp4lp_test_187`

**Why hard**: 4 distinct numbers: ['6', '3', '200', '30%']

**Text**: A volunteer organization transports voters to the polls on Election Day either by vans or cars. They have vans which can carry 6 people and cars which can carry 3 people.  They need to transport at least 200 voters to the polls. In addition, at most 30% of the vehicles can be vans. How many of each vehicle should be used to minimize the total numbe

**Numeric mentions**: `6`, `3`, `200`, `30%`

---

### 19. `nlp4lp_test_198`

**Why hard**: 8 distinct numbers: ['15', '20', '4', '7', '10', '$35']

**Text**: Lucy has a dog and she wants his food to be mixed.  In order to keep the dog healthy but also keep the food tasty, the mix needs to have a minimum of 15 units of calcium, 20 units of vitamin mix, and 20 units of protein. A regular brand costs $20 per bag and contains 4 units of calcium, 7 units of vitamin mix, and 10 units of protein. A premium bra

**Numeric mentions**: `15`, `20`, `4`, `7`, `10`, `$35`, `12`, `16`

---

### 20. `nlp4lp_test_209`

**Why hard**: 7 distinct numbers: ['$2800', '$2400', '3500', '20', '15', '6']

**Text**: A music company produces two types of digital keyboards, one is full-weighted and another is semi-weighted. Both keyboards are sold for $2800 and $2400 respectively. There are about 3500 oscillator chips available every day from which the full-weighted version requires 20 chips while the semi-weighted version requires 15 chips. The company has a to

**Numeric mentions**: `$2800`, `$2400`, `3500`, `20`, `15`, `6`, `1.2`

---

### 21. `nlp4lp_test_220`

**Why hard**: 6 distinct numbers: ['15', '8', '$2', '$3', '5', '$100']

**Text**: An amusement park has two types of games: throwing and climbing games. Throwing games attract 15 customers every hour and climbing games attract 8 customers every hour. Throwing games costs the amusement park $2 in prizes per hour whereas climbing games cost $3 in prizes per hour. Since throwing games yield the most profit, there must be at least t

**Numeric mentions**: `15`, `8`, `$2`, `$3`, `5`, `$100`

---

### 22. `nlp4lp_test_231`

**Why hard**: 9 distinct numbers: ['300', '15', '80', '200', '8', '20']

**Text**: A macro-counting fitness guru only eats salmon and eggs. Each bowl of salmon contains 300 calories, 15 grams of protein, and 80 mg of sodium. Each bowl of eggs contains 200 calories, 8 grams of protein, and 20 mg of sodium. Since the fitness guru has a limit to how many eggs he would like to eat, at most 40% of his meals can be eggs. The fitness gu

**Numeric mentions**: `300`, `15`, `80`, `200`, `8`, `20`, `40%`, `2000`, `90`

---

### 23. `nlp4lp_test_242`

**Why hard**: 4 distinct numbers: ['25', '45', '5', '750']

**Text**: A mask making company ships masks to their retail stores using small boxes and large boxes. A small box holds 25 masks whereas a large box holds 45 masks. Since small boxes are easier to stack and will be used first to stock stores, there must be at least three times as many small boxes as large boxes. Additionally, at least 5 large boxes must be u

**Numeric mentions**: `25`, `45`, `5`, `750`

---

### 24. `nlp4lp_test_253`

**Why hard**: 5 distinct numbers: ['60', '40', '$100', '$72', '10,000']

**Text**: A researcher is outsourcing annotations and has two options: a specialized third-party or a common third-party annotation company. The specialized company can annotate at a rate of 60 images per hour whereas the common company can annotate at a rate of 40 images per hour. However, the specialized company charges $100 per hour and the common company

**Numeric mentions**: `60`, `40`, `$100`, `$72`, `10,000`

---

### 25. `nlp4lp_test_264`

**Why hard**: 4 distinct numbers: ['30', '45', '300', '1500']

**Text**: Both chemical A and chemical B need to be added to a mixer for making bread. One unit of chemical A takes 30 seconds to be effective while one unit of chemical B takes 45 seconds to be effective. Because chemical A can be dangerous, there has to be at most a third as much chemical A as chemical B in the mixer. If there has to be at least 300 units 

**Numeric mentions**: `30`, `45`, `300`, `1500`

---

## Mixed Hard Cases (≥2 Slices)  (25 cases)

### 1. `nlp4lp_test_0`

**Why hard**: entity cues ['Mrs. Watson', 'Mrs. Watson'] with 5 numbers; lower cues ['minimum', 'at least'] AND upper cues ['at most']; 5 distinct numbers: ['$760000', '$0.50', '$1', '20%', '$20000']; total+per-unit language with numbers; percent cues ['20%'] + absolute values ['$760000', '$0.50']

**Text**: Mrs. Watson wants to invest in the real-estate market and has a total budget of at most $760000. She has two choices which include condos and detached houses. Each dollar invested in condos yields a $0.50 profit and each dollar invested in detached houses yields a $1 profit. A minimum of 20% of all money invested must be in condos, and at least $20

**Numeric mentions**: `$760000`, `$0.50`, `$1`, `20%`, `$20000`

---

### 2. `nlp4lp_test_21`

**Why hard**: lower cues ['greater than or equal'] AND upper cues ['maximum']; 8 distinct numbers: ['1', '2', '$5000', '$8500', '0', '8']

**Text**: A car manufacturer makes two versions of the same car, a regular model and a premium model. They make x1 regular models per day and x2 premium models per day. The profit per regular model is $5000 and the profit per premium model is $8500 (x1 and x2 are unknown values both greater than or equal to 0). The daily demand for these cars is limited to a

**Numeric mentions**: `1`, `2`, `$5000`, `$8500`, `0`, `8`, `6`, `12`

---

### 3. `nlp4lp_test_40`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']; 7 distinct numbers: ['50', '30', '20', '800', '700', '40%']; total+per-unit language with numbers

**Text**: A gem factory has two drills, a high intensity one and a low intensity one. Each day, the high intensity drill can process 50 gems and requires 50 units of water to dissipate heat. Each day, the low intensity drill can process 30 gems and requires 20 units of water to dissipate heat. Each day the factory must process 800 gems and they have availabl

**Numeric mentions**: `50`, `30`, `20`, `800`, `700`, `40%`, `10`

---

### 4. `nlp4lp_test_49`

**Why hard**: 6 distinct numbers: ['6', '$120', '10', '$250', '300', '$6500,']; total+per-unit language with numbers

**Text**: A city employs seasonal and permanent snow removers. A seasonal snow remover works 6 hours per shift and gets paid $120. A permanent snow remover works 10 hours per shift and gets paid $250. Currently the city needs 300 hours of snow remover labor after a heavy snowfall. If the city has a budget of $6500, how many of each type of worker should be h

**Numeric mentions**: `6`, `$120`, `10`, `$250`, `300`, `$6500,`

---

### 5. `nlp4lp_test_58`

**Why hard**: 7 distinct numbers: ['5', '4', '7', '6', '40%', '2500']; total+per-unit language with numbers; percent cues ['40%'] + absolute values ['2500', '3300']

**Text**: A bakery makes fiber supplemented brownies and lemon squares. Each brownie requires 5 units of chocolate mix and 4 units of fiber. Each lemon square requires 7 units of lemon mix and 6 units of fiber. Lemon squares sell much faster and thus the number of lemon squares made must be larger than the number of brownies made. However, to please all cust

**Numeric mentions**: `5`, `4`, `7`, `6`, `40%`, `2500`, `3300`

---

### 6. `nlp4lp_test_68`

**Why hard**: 7 distinct numbers: ['3', '6', '4', '7', '25%', '400']; total+per-unit language with numbers

**Text**: A crepe store sells chocolate and peanut butter crepes. A chocolate crepe requires 3 units of chocolate spread and 6 units of crepe mix. A peanut butter crepe requires 4 units of peanut butter spread and 7 units of crepe mix. Recently, the peanut butter crepe has been more popular and therefore the number of peanut butter crepes made must exceed th

**Numeric mentions**: `3`, `6`, `4`, `7`, `25%`, `400`, `450`

---

### 7. `nlp4lp_test_77`

**Why hard**: 5 distinct numbers: ['500', '750', '3', '20', '250000']; total+per-unit language with numbers

**Text**: A water company sells water in glass and plastic bottles. A glass bottle can hole 500 ml of water while a plastic bottle can hold 750 ml of water. Because most customer prefer plastic bottles, the number of plastic bottles must be at least 3 times the number of glass bottles. However, there must be at least 20 glass bottles. If the company has avai

**Numeric mentions**: `500`, `750`, `3`, `20`, `250000`

---

### 8. `nlp4lp_test_86`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']; 6 distinct numbers: ['30', '20', '4', '3', '200', '35']; total+per-unit language with numbers

**Text**: An office is buying printers for their headquarters, a premium model and regular model. The premium model can print 30 pages per minute while the regular model can print 20 pages per minute. In addition, the premium model requires 4 units of ink per minute while the regular model requires 3 units of ink per minute. The office wants to make sure tha

**Numeric mentions**: `30`, `20`, `4`, `3`, `200`, `35`

---

### 9. `nlp4lp_test_95`

**Why hard**: 9 distinct numbers: ['1', '2', '20', '15', '10', '30']; total+per-unit language with numbers

**Text**: A drug company is making allergy pills and fever reducing pills in two factories, factory 1 and factory 2. Factory 1 produces 20 allergy pills and 15 fever reducing pills per hour. Factory 2 produces 10 allergy pills and 30 fever reducing pills per hour.  Factory 1 is much more efficient and only requires 20 units of a rare compound while factory 2

**Numeric mentions**: `1`, `2`, `20`, `15`, `10`, `30`, `1000`, `700`, `600`

---

### 10. `nlp4lp_test_110`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']; 9 distinct numbers: ['400', '20', '100', '300', '10', '75']; percent cues ['30%'] + absolute values ['3000']

**Text**: A travelling salesman only eats ramen and fries. Each pack of ramen contains 400 calories, 20 grams of protein, and 100 mg of sodium. Each pack of fries contains 300 calories, 10 grams of protein, and 75 mg of sodium. Since fries are easier to eat while driving, at most 30% of his meals can be ramen. The salesman wants to ensure he eats at least 30

**Numeric mentions**: `400`, `20`, `100`, `300`, `10`, `75`, `30%`, `3000`, `80`

---

### 11. `nlp4lp_test_123`

**Why hard**: lower cues ['at least', 'at least'] AND upper cues ['at most']; 6 distinct numbers: ['200', '40', '250', '50', '3', '10000']; total+per-unit language with numbers

**Text**: A competitive eater challenges himself to eat slices of cheesecake and caramel cake. Each slice of cheesecake contains 200 calories and 40 grams of sugar. Each slice of caramel cake contains 250 calories and 50 grams of sugar. He prefers cheesecake and decides to eat at least 3 times as many slices of cheesecake as caramel cake. However, he must al

**Numeric mentions**: `200`, `40`, `250`, `50`, `3`, `10000`

---

### 12. `nlp4lp_test_134`

**Why hard**: 9 distinct numbers: ['1', '2', '35', '12', '50', '30']; total+per-unit language with numbers

**Text**: A drug company is making pain killers and sleeping pills using two processes, process 1 and process 2. Process 1 produces 35 units of pain killers and 12 units of sleeping pills per hour. Process 2 produces 50 units of pain killers and 30 units of sleeping pills per hour. Process 1 requires 50 units of preliminary material while process 2 requires 

**Numeric mentions**: `1`, `2`, `35`, `12`, `50`, `30`, `60`, `2000`, `1200`

---

### 13. `nlp4lp_test_145`

**Why hard**: lower cues ['minimum', 'minimum'] AND upper cues ['cannot exceed']; 3 distinct numbers: ['50', '100', '2000']

**Text**: A shoe company supplies shoes to stores via vans and trucks. A van can transport 50 pairs of shoes while a truck can transport 100 pairs of shoes. The company must supply a minimum of 2000 pairs of shoes around the city. Since most stores are small, the number of trucks used cannot exceed the number of vans used.  Find the minimum number of vans th

**Numeric mentions**: `50`, `100`, `2000`

---

### 14. `nlp4lp_test_154`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']; 7 distinct numbers: ['8', '3', '5', '2', '30%', '20']

**Text**: A meal service company delivers meals to customers either on electric bikes or scooters. A bike can hold 8 meals and requires 3 units of charge. A scooter can hold 5 meals and requires 2 units of charge. Since the city is more friendly towards scooters, at most 30% of the electric vehicles can be bikes and at least 20 scooters must be used. If the 

**Numeric mentions**: `8`, `3`, `5`, `2`, `30%`, `20`, `200`

---

### 15. `nlp4lp_test_164`

**Why hard**: lower cues ['at least'] AND upper cues ['maximum']; 6 distinct numbers: ['5', '2', '8', '200', '30%', '10']; total+per-unit language with numbers

**Text**: A toy store hires seasonal and full-time volunteers to deliver gifts and gives them points for service. A seasonal volunteer can deliver 5 gifts and gets 2 points. A full-time volunteer can deliver 8 gifts and gets 5 points. The store can only give out 200 points. In addition, a maximum of 30% of the volunteers can be seasonal and at least 10 must 

**Numeric mentions**: `5`, `2`, `8`, `200`, `30%`, `10`

---

### 16. `nlp4lp_test_174`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']; 7 distinct numbers: ['7', '5', '10', '6', '1500', '50']; percent cues ['60%'] + absolute values ['1500']

**Text**: A dog school trains labradors and golden retrievers to deliver newspaper. A labrador can deliver 7 newspapers at a time and requires 5 small bone treats for service. A golden retriever can deliver 10 newspapers at a time and requires 6 small bone treats per service. The school only has 1500 small bone treats available. In addition, at least 50 gold

**Numeric mentions**: `7`, `5`, `10`, `6`, `1500`, `50`, `60%`

---

### 17. `nlp4lp_test_182`

**Why hard**: lower cues ['At least'] AND upper cues ['at most']; 5 distinct numbers: ['4', '10', '20', '30', '300']; total+per-unit language with numbers

**Text**: Employees have the option of car-pooling to work or taking the company bus. A car can take 4 employees and produces 10 units of pollution, while a bus can take 20 employees and produces 30 units of pollution. At least 300 employees need to be transported and at most 4 buses can be used. How many of each type of transport should be taken to minimize

**Numeric mentions**: `4`, `10`, `20`, `30`, `300`

---

### 18. `nlp4lp_test_199`

**Why hard**: entity cues ['Maple Oil'] with 11 numbers; 9 distinct numbers: ['$550,', '$750,', '$950', '3', '6', '2']

**Text**: Maple Oil processes three types of crude oil: light oil, non-sticky oil and heavy oil. Each tank of light oil produces a net revenue of $550, each tank of non-sticky oil produces a net revenue of $750, and each tank of heavy oil produces a net revenue of $950. To process a tank of light oil, 3 units of compound A and 3 units of compound B are requi

**Numeric mentions**: `$550,`, `$750,`, `$950`, `3`, `6`, `2`, `9`, `250`, `150`

---

### 19. `nlp4lp_test_210`

**Why hard**: entity cues ['Platinum Database'] with 6 numbers; 6 distinct numbers: ['$550', '$2000', '300', '$450', '$1200', '$400000,']; total+per-unit language with numbers

**Text**: Platinum Database sells two types of subscription software packages: a personal license and a commercial license which will cost $550 and $2000 to generate respectively. The marketing department estimates that they can sell at most 300 licenses for both versions combined a month. The profit per personal license is $450 and the profit per commercial

**Numeric mentions**: `$550`, `$2000`, `300`, `$450`, `$1200`, `$400000,`

---

### 20. `nlp4lp_test_218`

**Why hard**: 7 distinct numbers: ['12', '150', '17', '250', '30%', '7']; total+per-unit language with numbers; percent cues ['30%'] + absolute values ['3400']

**Text**: A strata-management company is looking into purchasing two types of air conditioners, a low-power and a high-power model. A low-powered air conditioner can cool down 12 housing units and uses 150 units of electricity every day. A high-power model can cool down 17 housing units and uses 250 units of electricity every day. Since the low-powered model

**Numeric mentions**: `12`, `150`, `17`, `250`, `30%`, `7`, `3400`

---

### 21. `nlp4lp_test_228`

**Why hard**: 4 distinct numbers: ['12', '18', '400', '70%']; total+per-unit language with numbers

**Text**: A party organizer needs to transport party goers either by limousine or bus. Limousines can carry 12 people and buses can carry 18 people. They need to transport at least 400 people. Because limousines are more attractive, at least 70% of the vehicles must be limousines. How many of each type of vehicle should be used to minimize the total number o

**Numeric mentions**: `12`, `18`, `400`, `70%`

---

### 22. `nlp4lp_test_239`

**Why hard**: 4 distinct numbers: ['250', '1000', '1000000', '100']; total+per-unit language with numbers

**Text**: A soda company sells soda in two types of containers: cans and glass bottles. A can holds 250 ml of soda whereas a bottle holds 1000 ml of soda. The soda company needs to bottle at least 1000000 ml of soda every day and due to the ability to sell soda cans as packs, there must be at least three times more cans than glass bottles. However, because o

**Numeric mentions**: `250`, `1000`, `1000000`, `100`

---

### 23. `nlp4lp_test_250`

**Why hard**: lower cues ['at least'] AND upper cues ['at most']; 9 distinct numbers: ['30', '4', '$100', '70', '15', '$225']; total+per-unit language with numbers; percent cues ['25%'] + absolute values ['$100', '$225']

**Text**: A taxi company will purchase vehicles to rent to their drivers. They are interested in purchasing either motorcycles or sedans. A motorcycle can transport 30 people, produces 4 units of pollution, and earns the taxi company $100 per shift. A sedan can transport 70 people, produces 15 units of pollution and earns the company $225 per shift. Because 

**Numeric mentions**: `30`, `4`, `$100`, `70`, `15`, `$225`, `25%`, `200`, `1200`

---

### 24. `nlp4lp_test_261`

**Why hard**: lower cues ['must be at least'] AND upper cues ['at most']; 7 distinct numbers: ['3', '5', '6', '100', '400', '530']; total+per-unit language with numbers

**Text**: A hospital prepares batches of medication patches and anti-biotic creams. Each medication patch requires 3 minutes to prepare and 5 units of materials. Each anti-biotic cream requires 5 minutes to prepare and 6 units of materials. Since anti-biotic creams are used more often, there must be at least twice as many anti-biotic creams as medication pat

**Numeric mentions**: `3`, `5`, `6`, `100`, `400`, `530`, `2`

---

### 25. `nlp4lp_test_271`

**Why hard**: lower cues ['at least', 'at least'] AND upper cues ['at most']; 3 distinct numbers: ['250', '400', '1.5']; total+per-unit language with numbers

**Text**: A chemical company uses two tests, a salinity test and a pH test. Each unit of the salinity test requires three probes. Whereas each unit of the pH test requires two probes. The chemical company must perform at least 250 pH tests. In total, at least 400 tests must be performed. Further, because of the importance of the salinity test, there must be 

**Numeric mentions**: `250`, `400`, `1.5`

---

