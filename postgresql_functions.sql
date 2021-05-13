--------------------------------------------------------------------------------------------------------------------
--
-- File:        postgresql_functions.sql
-- Description: Load saved Machine learning model inside postgresql, and use trigger to predict in case of insert.
-- Author:      Mostafa Mamdouh
-- Created:     Mon May 10 20:23:43 PDT 2021
--
--------------------------------------------------------------------------------------------------------------------


-- Activate python3 in postgresql
create extension plpython3u;

-- Know python version used by postgresql (usually your default python)
create or replace function get_python_version()
returns text
AS $$
    import sys
    return(sys.version)
$$ language 'plpython3u';

-- select get_python_version();


-- plpython3u function to load the model and predict
create or replace function call_model(sepal_length numeric, sepal_width numeric, petal_length numeric, petal_width numeric)
returns character varying
AS $$
    from joblib import load
    X_test = [[sepal_length, sepal_width , petal_length, petal_width]]
    path = r'/model/file/path'
    classifier = load(path)
    y_pred_encoded = int(classifier.predict(X_test))
    y_pred = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}.get(y_pred_encoded, 'model error') 
    return(y_pred)
$$ language 'plpython3u';

-- select call_model(3, 3, 3, 3);


-- plpgsql function to call the plpython function and load the model
create or replace function predict()
returns trigger as
$BODY$
begin
	new.species = call_model(new.sepal_length, new.sepal_width, new.petal_length, new.petal_width);
	return new;
end;
$BODY$
language plpgsql;


-- trigger in case of insert
create trigger prediction_triger
before insert on iris
for each row
execute procedure predict();


/*
-- test all
-- insert to iris table
insert into iris (sepal_length, sepal_width, petal_length, petal_width) 
values (3, 3, 3, 3);

-- select from iris table
select * from iris;
*/

