# Machine_learning_using_postgresql_plpython

- Machine learning using Postgresql plpython is a solution to embed Python inside Postgresql for Machine learning's sake.
- It is a combination of the power of Postgresql, things like Triggers, and Plpgsql function, together with the power of Python and its ML ability.
- So in the case of insert, a Trigger is fired, and call Plpython function inside Postgresql to predict, Which means everything is embedded inside Postgresql internally without calling outside python script hence avoids the need to set up a connection between your Database and Python.
- The script is applied to Iris dataset but you can use the concept to apply it to any dataset.

## Installation

See installation_guide, it contains steps.txt, and requirements.txt

## Usage

##### Python
```bash
python iris_classifier.py
python iris_predictor.py
```

##### Postgresql
```sql
insert into iris (sepal_length, sepal_width, petal_length, petal_width) values (3, 3, 3, 3);
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


