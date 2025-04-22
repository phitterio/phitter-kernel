* [phitter](modules.html)
* [phitter package](phitter.html)
* [phitter.simulation package](phitter.simulation.html)
* phitter.simulation.queueing\_simulation package
* [View page source](_sources/phitter.simulation.queueing_simulation.rst.txt)

---

# phitter.simulation.queueing\_simulation package

## Submodules

## phitter.simulation.queueing\_simulation.queueing\_simulation module

*class* phitter.simulation.queueing\_simulation.queueing\_simulation.QueueingSimulation(*a*, *a\_parameters*, *s*, *s\_parameters*, *c*, *k=inf*, *n=inf*, *d='FIFO'*, *pbs\_distribution=None*, *pbs\_parameters=None*)
:   Bases: `object`

## average\_elements\_queue()
Average elements in queue

#### Returns
Average elements in queue

#### Return type
float

## average\_elements\_system()
Average elements in system

#### Returns
Average elements in system

#### Return type
float

## average\_time\_queue()
Average time in queue

#### Returns
Average time in queue

#### Return type
float

## average\_time\_service()
Average time in service

#### Returns
Average time in service

#### Return type
float

## average\_time\_system()
Average time in system

#### Returns
Average time in system

#### Return type
float

## confidence\_interval\_metrics(*simulation\_time=inf*, *confidence\_level=0.95*, *replications=30*)
Generate a confidence interval for probabilities and metrics.

#### Parameters
* **simulation\_time** (*int**,* *optional*) – Simulation time. Defaults to float(“Inf)
- * **confidence\_level** (*int**,* *optional*) – Confidence level for the confidence interval for all the metrics and probabilities. Defaults to 0.95.
- * **replications** (*int**,* *optional*) – Number of samples of simulations to create. Defaults to 30.

#### Returns
Returns probabilities and metrics dataframe with confidene interval for all metrics.

#### Return type
tuple[pd.DataFrame, pd.DataFrame]

## elements\_prob(*bins=50000*)
Creates the probability for each number of elements. Example: Probability to be 0, prob. to be 1, prob. to be 2… depending on simulation values

#### Parameters
**bins** (*int**,* *optional*) – Number of intervals to determine the probability to be in each stage. Defaults to 50000.

#### Returns
Element and probability result

#### Return type
dict

## metrics\_summary()
Returns the summary of the following metrics: Average Time in System, Average Time in Queue, Average Time in Service, Std. Dev. Time in System, Std. Dev. Time in Queue, Std. Dev. Time in Service, Average Elements in System, Average Elements in Queue, Probability to join the System, Probability to finish after Time, Probability to Wait in Line

#### Returns
Returns dataframe with all the information

#### Return type
pd.DataFrame

## no\_clients\_prob()
Probability of no having clients

#### Returns
No clients probability

#### Return type
float

## number\_elements\_prob(*number*, *prob\_type*)
Calculates the probability Exact, less or equals or greater or equals.

#### Parameters
* **number** (*int*) – Number that we want to identify the different probabilities
- * **prob\_type** (*str*) – Could be one of the following options: ‘exact\_value’, ‘greater\_equals’, ‘less\_equals’

#### Returns
Probability of the number of elements

#### Return type
float

## number\_probability\_summary()
Returns the probability for each element. The probability is Exact, less or equals or greater or equals; represented in each column.

#### Returns
Dataframe with all the needed probabilities for each element.

#### Return type
pd.DataFrame

## probability\_to\_finish\_after\_time()
Probability to finish after time

#### Returns
Probability to finish after time

#### Return type
float

## probability\_to\_join\_system()
Probability to join the system

#### Returns
Probability to join the system

#### Return type
float

## probability\_to\_wait\_in\_line()
Probability to wait in the queue

#### Returns
Probability to wait in the queue

#### Return type
float

## run(*simulation\_time=inf*)
Simulation of any queueing model.

#### Parameters
* **simulation\_time** (*float**,* *optional*) – This variable defines the total duration of the simulation. It sets the length of time over which the simulation will model the system’s behavior. Defaults to float(“inf”)
- * **number\_of\_simulations** (*int**,* *optional*) – Number of simulations of the process. Can also be considered as the number of days or number of times you want to simulate your scenario. Defaults to 1.

#### Returns
[description]

#### Return type
tuple

## servers\_utilization()
Determine the server utilization according to the simulation result

#### Returns
Utilization of all servers, you can find the server number in the rows

#### Return type
pd.DataFrame

## standard\_deviation\_time\_queue()
Standard Deviation time in queue

#### Returns
Standard Deviation time in queue

#### Return type
float

## standard\_deviation\_time\_service()
Standard Deviation time in service

#### Returns
Standard Deviation time in service

#### Return type
float

## standard\_deviation\_time\_system()
Standard Deviation time in system

#### Returns
Standard Deviation time in system

#### Return type
float

## system\_utilization()
Returns system utilization according to simulation

#### Returns
System Utilization

#### Return type
float

## to\_csv(*file\_name*, *index=True*)
Simulation results to CVS

#### Parameters
* **file\_name** (*str*) – File Name to add to the CSV file. You should include “.csv” at the end of your file
- * **index** (*bool**,* *optional*) – Defaults to True. Add index in CSV file.

#### Return type
`None`

## to\_excel(*file\_name*, *sheet\_name='Sheet1'*, *index=True*)
Simulation results to Excel File

#### Parameters
* **file\_name** (*str*) – File Name to add to the Excel file. You should include “.xlsx” at the end of your file
- * **index** (*bool**,* *optional*) – Defaults to True. Add index in Excel file.

#### Return type
`None`

## Module contents

[Previous](phitter.simulation.process_simulation.html "phitter.simulation.process_simulation package")

---

© Copyright .

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).