# Representative Failure Examples

> **Methodology note:** Examples are selected programmatically from per-instance
> diagnostics for the TF-IDF baseline, orig variant.
> Reasons labelled *likely* are inferred from measurable signals;
> reasons labelled *measured* are directly observable from data.

---

## [schema_hit_but_not_inst_ready] Example 1

**Query ID:** `nlp4lp_test_10`  
**Reason:** likely: type mismatch or incomplete coverage despite correct schema  
**Schema hit:** 1  
**Coverage:** 0.7500  
**TypeMatch:** 1.0000  
**InstReady:** 0  
**n_expected_scalar:** 8  
**n_filled:** 6  
**Numeric mentions in query:** 7  

**Query (full):**
> A bakery uses a stand-mixer and a slow bake oven to make bread and cookies. Each machine can run for at most 3000 hours per year. To bake a loaf of bread takes 1 hour in the stand mixer and 3 hours in the oven. A batch of cookies requires 0.5 hours in the mixer and 1 hour in the oven. The profit per loaf of bread is $5 and the profit per batch of cookies is $3. How should the bakery operate to maximize total profit?

**Gold schema text (preview):**
> A bakery uses a stand mixer with MixerMaximumHours available operating hours per year and an oven with OvenMaximumHours available operating hours per year. Producing one loaf of bread requires BreadMi

---

## [schema_hit_but_not_inst_ready] Example 2

**Query ID:** `nlp4lp_test_14`  
**Reason:** likely: type mismatch or incomplete coverage despite correct schema  
**Schema hit:** 1  
**Coverage:** 0.2000  
**TypeMatch:** 1.0000  
**InstReady:** 0  
**n_expected_scalar:** 5  
**n_filled:** 1  
**Numeric mentions in query:** 4  

**Query (full):**
> My family has decided to invest in real state for the first time. Currently, they have $600,000 to invest, some in apartments and the rest in townhouses. The money invested in apartments must not be greater than $200,000. They have decided that the money invested in apartments must be at least a half as much as that in townhouses.  If the apartments earn 10%, and the townhouses earn 15%, how much money should they invest in each to maximize profit?

**Gold schema text (preview):**
> A family has a TotalInvestment to allocate between apartments and townhouses. The investment in apartments must not exceed MaxInvestmentApartments and must be at least MinInvestmentRatio times the inv

---

## [schema_hit_but_not_inst_ready] Example 3

**Query ID:** `nlp4lp_test_17`  
**Reason:** likely: type mismatch or incomplete coverage despite correct schema  
**Schema hit:** 1  
**Coverage:** 0.6667  
**TypeMatch:** 0.8333  
**InstReady:** 0  
**n_expected_scalar:** 9  
**n_filled:** 6  
**Numeric mentions in query:** 9  

**Query (full):**
> An electronics store wants to optimize how many phones and laptops are enough to keep in inventory. A phone will earn the store $120 in profits, and a laptop will earn $40. A phone requires 1 sq ft of floor space, whereas a laptop requires 4 sq ft. In total, 400 sq ft of floor space is available. The store stocks only phones and laptops. Corporate has required that at least 80% of all appliances in stock be laptops. Finally, a phone costs $400 for the store, and a laptop, $100. The store wants to spend at most $6000. Formulate an LP that can be used to maximize the store's profit.

**Gold schema text (preview):**
> An electronics store seeks to determine the optimal quantities of two types of products to maintain in inventory to maximize profit. Each unit of the first product yields ProfitPhone in profit, while 

---

## [type_mismatch] Example 1

**Query ID:** `nlp4lp_test_21`  
**Reason:** likely: wrong type assigned to slot (float vs integer, percent vs absolute)  
**Schema hit:** 1  
**Coverage:** 1.0000  
**TypeMatch:** 0.6000  
**InstReady:** 0  
**n_expected_scalar:** 5  
**n_filled:** 5  
**Numeric mentions in query:** 6  

**Query (full):**
> A car manufacturer makes two versions of the same car, a regular model and a premium model. They make x1 regular models per day and x2 premium models per day. The profit per regular model is $5000 and the profit per premium model is $8500 (x1 and x2 are unknown values both greater than or equal to 0). The daily demand for these cars is limited to and most 8 regular models and 6 premium models. In addition, the manufacturer can make a maximum of 12 cars of either type per day. How many cars of each model should the manufacturer make in order to maximize profit?

**Gold schema text (preview):**
> The manufacturer aims to maximize total profit, calculated as ProfitRegular multiplied by the number of regular models produced (x₁) plus ProfitPremium multiplied by the number of premium models produ

---

## [type_mismatch] Example 2

**Query ID:** `nlp4lp_test_41`  
**Reason:** likely: wrong type assigned to slot (float vs integer, percent vs absolute)  
**Schema hit:** 1  
**Coverage:** 1.0000  
**TypeMatch:** 0.3333  
**InstReady:** 0  
**n_expected_scalar:** 3  
**n_filled:** 3  
**Numeric mentions in query:** 6  

**Query (full):**
> A sandwich company can open two types of stores, a dine-in place and a food-truck. A dine-in place can make 100 sandwiches per day and requires 8 employees to operate. A food-truck can make 50 sandwiches per day and requires 3 employees to operate. The company must make at least 500 sandwiches per day but they only have available 35 employees. How many of each type of store should the company open to minimize the total number of stores?

**Gold schema text (preview):**
> A sandwich company can open NumStoreTypes types of stores. Each store type produces SandwichesPerStoreType sandwiches per day and requires EmployeesPerStoreType employees to operate. The company must 

---

## [type_mismatch] Example 3

**Query ID:** `nlp4lp_test_46`  
**Reason:** likely: wrong type assigned to slot (float vs integer, percent vs absolute)  
**Schema hit:** 1  
**Coverage:** 1.0000  
**TypeMatch:** 0.6667  
**InstReady:** 0  
**n_expected_scalar:** 6  
**n_filled:** 6  
**Numeric mentions in query:** 6  

**Query (full):**
> An amusement park is installing ticket machines, both cash-based and card-only machines. A cash-based machine can process 20 people per hour while a card-only machine can process 30 people per hour. The cash-based machine needs 4 rolls of paper per hour while the card-only machine requires 5 rolls of paper per hour. The amusement park needs to be able to process at least 500 people per hour but can use at most 90 rolls of paper per hour. Since most people prefer to pay by cash, the number of card-only machines must not exceed the number of cash-based machines. How many of each machine should be bought to minimize the total number of machines in the park?

**Gold schema text (preview):**
> An amusement park is installing cash-based machines and card-only machines. A cash-based machine can process CashMachineProcessingRate people per hour, while a card-only machine can process CardMachin

---

## [likely_slot_disambiguation_failure] Example 1

**Query ID:** `nlp4lp_test_41`  
**Reason:** likely: coverage acceptable but most types wrong; probable slot/type confusion  
**Schema hit:** 1  
**Coverage:** 1.0000  
**TypeMatch:** 0.3333  
**InstReady:** 0  
**n_expected_scalar:** 3  
**n_filled:** 3  
**Numeric mentions in query:** 6  

**Query (full):**
> A sandwich company can open two types of stores, a dine-in place and a food-truck. A dine-in place can make 100 sandwiches per day and requires 8 employees to operate. A food-truck can make 50 sandwiches per day and requires 3 employees to operate. The company must make at least 500 sandwiches per day but they only have available 35 employees. How many of each type of store should the company open to minimize the total number of stores?

**Gold schema text (preview):**
> A sandwich company can open NumStoreTypes types of stores. Each store type produces SandwichesPerStoreType sandwiches per day and requires EmployeesPerStoreType employees to operate. The company must 

---

## [likely_slot_disambiguation_failure] Example 2

**Query ID:** `nlp4lp_test_76`  
**Reason:** likely: coverage acceptable but most types wrong; probable slot/type confusion  
**Schema hit:** 1  
**Coverage:** 1.0000  
**TypeMatch:** 0.0000  
**InstReady:** 0  
**n_expected_scalar:** 6  
**n_filled:** 6  
**Numeric mentions in query:** 6  

**Query (full):**
> A butcher shop is buying meat slicers for their shop, a manual and automatic slicer. The manual slicer can cut 5 slices per minute while the automatic slicer can cut 8 slices per minute. In addition, to make sure all the parts operate smoothly, the manual slicer requires 3 units of grease per minute while the automatic slicer requires 6 units of grease per minute. Since the automatic slicer does not need someone to operate it, the number of manual slicers must be less than the number of automatic slicers. In addition, the butcher shop needs to be able to cut at least 50 slices per minute but can use at most 35 units of grease per minute. How many of each slicer should the butcher shop buy to minimize the total number of slicers in the shop?

**Gold schema text (preview):**
> The butcher shop purchases a certain number of Manual and Automatic slicers. Each Manual slicer can cut ManualSliceRate slices per minute and requires ManualGreaseRate units of grease per minute. Each

---

## [likely_slot_disambiguation_failure] Example 3

**Query ID:** `nlp4lp_test_109`  
**Reason:** likely: coverage acceptable but most types wrong; probable slot/type confusion  
**Schema hit:** 1  
**Coverage:** 1.0000  
**TypeMatch:** 0.2500  
**InstReady:** 0  
**n_expected_scalar:** 8  
**n_filled:** 8  
**Numeric mentions in query:** 14  

**Query (full):**
> A patient can be hooked up to two machines to have medicine delivered, machine 1 and machine 2. Machine 1 delivers 0.5 units of medicine to the heart per minute and 0.8 units of medicine per minute to the brain. Machine 2 delivers 0.3 units of medicine per minute to the heart and 1 unit of medicine per minute to the brain. In addition however, machine 1 creates 0.3 units of waste per minute while machine 2 creates 0.5 units of waste per minute. If at most 8 units of medicine can be received by the heart and at least 4 units of medicine should be received by the brain, how many minutes should each machine be used to minimize the total amount of waste produced?

**Gold schema text (preview):**
> A patient can be connected to two machines, Machine1 and Machine2, for certain durations. Machine1 delivers Machine1HeartDeliveryRate units of medicine to the heart per minute and Machine1BrainDeliver

---

## [schema_miss] Example 1

**Query ID:** `nlp4lp_test_4`  
**Reason:** measured: retrieval returned wrong schema; downstream grounding irrelevant  
**Schema hit:** 0  
**Coverage:** 0.0000  
**TypeMatch:** 0.0000  
**InstReady:** 0  
**n_expected_scalar:** 0  
**n_filled:** 0  
**Numeric mentions in query:** 5  

**Query (full):**
> A store employs senior citizens who earn $500 per week and young adults who earn $750 per week. The store must keep the weekly wage bill below $30000. On any day, the store requires at least 50 workers, of whom at least 10 must be young adults. To ensure the store runs smoothly, the number of young adults should be at least a third the number of senior citizens. Formulate a LP to minimize the wage bill.

**Gold schema text (preview):**
> Minimize (SeniorWage × NumberOfSeniorCitizens + YoungAdultWage × NumberOfYoungAdults) subject to (SeniorWage × NumberOfSeniorCitizens + YoungAdultWage × NumberOfYoungAdults ≤ MaxWeeklyWageBill), (Numb

**Predicted schema text (preview):**
> A sandwich company can open NumStoreTypes types of stores. Each store type produces SandwichesPerStoreType sandwiches per day and requires EmployeesPerStoreType employees to operate. The company must 

---

## [schema_miss] Example 2

**Query ID:** `nlp4lp_test_12`  
**Reason:** measured: retrieval returned wrong schema; downstream grounding irrelevant  
**Schema hit:** 0  
**Coverage:** 0.0000  
**TypeMatch:** 0.0000  
**InstReady:** 0  
**n_expected_scalar:** 0  
**n_filled:** 0  
**Numeric mentions in query:** 8  

**Query (full):**
> A souvenir shop makes wooden elephants and tigers with plastic ornaments. Each elephant requires 50 grams of wood and 20 grams of plastic. Each tiger requires 40 grams of wood and 30 grams of plastic. In a week, 5000 grams of wood and 4000 grams of plastic are available. The profit per elephant sold is $5 and the profit per tiger sold is $4. How many of each should be made in order to maximize profit?

**Gold schema text (preview):**
> A souvenir shop produces NumProducts different products using NumResources different resources. Each product has a profit defined by Profit and requires specific amounts of resources as specified by R

**Predicted schema text (preview):**
> A clinical firm operates two factories, northern and western. The firm decides the number of hours to run each factory. The northern factory produces NorthernFactoryAntiItchRate grams of anti-itch inj

---

## [schema_miss] Example 3

**Query ID:** `nlp4lp_test_24`  
**Reason:** measured: retrieval returned wrong schema; downstream grounding irrelevant  
**Schema hit:** 0  
**Coverage:** 0.0000  
**TypeMatch:** 0.0000  
**InstReady:** 0  
**n_expected_scalar:** 0  
**n_filled:** 0  
**Numeric mentions in query:** 8  

**Query (full):**
> Sleep inducing medicine and anti-inflammatory medicine is found in two pills, pill A and pill B. One pill A contains 3 units of sleep inducing medicine and 5 units of anti-inflammatory medicine. One pill B contains 6 units of sleep-inducing medicine and 1 unit of anti-inflammatory medicine. The cost per pill for pill A is $4 and the cost per pill for pill B is $5. A patient must consume these two pills to get at least 40 units of sleep-inducing medicine and 50 units of anti-inflammatory medicine. Formulate a LP to minimize the cost for the patient.

**Gold schema text (preview):**
> A patient selects non-negative quantities of each of the NumPillTypes pill types. Each pill type provides specific amounts of each of the NumMedicineTypes medicine types as defined by AmountPerPill. T

**Predicted schema text (preview):**
> A pharmaceutical company has 800 units of painkiller medicine and makes daytime and nighttime painkiller. A daytime pill has 6 units of painkiller medicine and 2 units of sleep medicine. A nighttime p
