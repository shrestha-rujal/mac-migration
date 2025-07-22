from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import corr

from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler

df=spark.read.csv('Linear_regression_dataset.csv',inferSchema=True,header=True)

df.select(corr('var_1','output')).show()

vec_assmebler=VectorAssembler(inputCols=['var_1', 'var_2', 'var_3', 'var_4', 'var_5'],outputCol='features')

features_df=vec_assmebler.transform(df)

train_df,test_df=model_df.randomSplit([0.7,0.3])

lin_Reg=LinearRegression(labelCol='output')

lr_model=lin_Reg.fit(train_df)

lr_model.intercept
