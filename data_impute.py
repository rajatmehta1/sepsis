from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

spark = SparkSession.builder.getOrCreate()

sqlContext = SQLContext(spark.sparkContext)

phyDF = sqlContext.read.load('file:///C:/mystuff/gatech/project/training/*.psv',format='csv',sep='|',inferSchema='true',header='true')


phyDF.show(1)

phyDF.printSchema()


# In[55]:


sqlContext.registerDataFrameAsTable(phyDF,"patients_table")


# In[56]:


vitalsDF = sqlContext.sql("select HR,O2Sat,Temp,SBP,MAP,DBP,Resp,EtCO2,SepsisLabel from patients_table")


# In[57]:


vitalsDF.show(2)


# In[58]:


vitalsDF.count()


# In[59]:


vitalsWithNanDF = sqlContext.sql("select HR,O2Sat,Temp,SBP,MAP,DBP,Resp,EtCO2,SepsisLabel from patients_table where " +
                                 "( isNan(HR) = true or isNan(O2Sat) = true or isNan(Temp) = true or isNan(SBP) = true or isNan(MAP) = true or isNan(DBP) = true or isNan(Resp) = true or isNan(EtCO2) = true or isNan(SepsisLabel) = true)")


# In[60]:


vitalsWithNanDF.show(5)


# In[61]:


emptyVitalsDF = sqlContext.sql("select HR,O2Sat,Temp,SBP,MAP,DBP,Resp,EtCO2,SepsisLabel from patients_table where " +
                                 "( isNan(HR) = true and isNan(O2Sat) = true and isNan(Temp) = true and isNan(SBP) = true and isNan(MAP) = true and isNan(DBP) = true and isNan(Resp) = true and isNan(EtCO2) = true)")


# In[62]:


emptyVitalsDF.show(2)


# In[63]:


vitalsWithoutEmptyDF = vitalsDF.subtract(emptyVitalsDF)


# In[64]:


vitalsWithoutEmptyDF.count()


# In[65]:


vitalsWithoutEmptyDF.createOrReplaceTempView("vitals_tbl")


# In[66]:


sqlContext.sql("select * from vitals_tbl").show(2)


# In[67]:


from pyspark.sql import functions as F


# In[68]:


vitals_main_df = vitalsWithoutEmptyDF.withColumn('idx', F.monotonically_increasing_id())


# In[69]:


vitals_main_df.show(3)


# In[70]:


vitals_main_df.createOrReplaceTempView("vitals_tbl")


# In[71]:


sqlContext.sql("select HR,O2Sat,Temp,SBP,MAP,DBP,Resp,EtCO2,float(SepsisLabel) from vitals_tbl").show()


# In[72]:


vitals_with_missing_df = sqlContext.sql("select HR,O2Sat,Temp,SBP,MAP,DBP,Resp,EtCO2,float(SepsisLabel) from vitals_tbl")


# In[73]:


from pyspark.ml.feature import Imputer


# In[74]:


imputer = Imputer(inputCols=["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","SepsisLabel"], outputCols=["o_HR","o_O2Sat","o_Temp","o_SBP","o_MAP","o_DBP","o_Resp","o_EtCO2","o_SepsisLabel"])


# In[75]:


model = imputer.fit(vitals_with_missing_df)


# In[76]:


vitals_with_clean_df = model.transform(vitals_with_missing_df)


# In[77]:


vitals_with_clean_df.show(3)


# In[78]:


vitals_with_clean_df.createOrReplaceTempView('clean_vitals_data')


# In[79]:


vitals_data_df = sqlContext.sql("select o_HR as HR,o_O2Sat as O2Sat,o_Temp as Temp,o_SBP as SBP,o_MAP as MAP,o_DBP as DBP,o_Resp as Resp,o_EtCO2 as EtCO2,o_SepsisLabel as SepsisLabel from clean_vitals_data")


# In[80]:


vitals_data_df.show(5)


# In[83]:


vitals_data_df.repartition(1).write.csv('file:///C:/mystuff/hadoop/bin/output/out52.csv',header='true')





