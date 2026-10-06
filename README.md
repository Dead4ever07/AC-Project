# AC-Project


1. The final regular season ranking of teams 
1. The set of teams that will change coaches

### Analysis of requirements with the end user
- What would be an acceptable accuracy?
    - We can use as a baseline the past year(medelo de persistencia) and then calculate how far from the actual value we are
    - If it is worst than the base line it is bad
    - For a more formal metric we can create a distance function and do a (dist(future, predicted) - dist(future, past))² and check its value
- How much should the model take to run?
    - No constraints on that manner

- What is a change of coaches?
    - We are predicting changes from one season to another.

- Should the result of the rankings of the season be a probability matrix or a list from 1 to N?
    - The final result should be a the list of classifications in absolute form.

- The same for the output on the coaches, probability of changing or absolute value?
    - The result should also be a list of what coaches will change in absolute form.

# Definition of business goals 

- Anticipate team performance by predicting the final regular season ranking of all teams.
- Forecast organizational instability by identifying the set of teams that will change coaches.
- Address a third predictive objective that is currently to be announced (TBA).
- Operate under the strict temporal constraint that all predictions must be made the day before the first game of the new season.

# Translation of business goals into data mining goals 

- Ranking/Regression/Classification problem that requires leveraging past data.
- Binary classification problem that requires analysis of past statistics of the coaches/teams.
- Data mining process must success fully capture the relations between coachs and teams. 