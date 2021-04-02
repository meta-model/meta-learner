<h3>Code Organization:</h3>

<ul>
  <li>Train_meta_model_baselines_explicit.py is the training of the meta-learners, single recommendation algorithms and stacking models. Models will be saved in directory models. Test csv files for meta and stacking learned algorithms will be generated.</li>
  <li>measure_metrics_explicit.py is used to generate Top N recommendations and calculate values of the metrics.</li>
  <li>The other files represent the implementation of recommendation algorithms.</li>
</ul>

<h3>Requirement:</h3>
<ul>
  <li>pandas>=1.1.0</li>
  <li>numpy>=1.19.5</li>
  <li>scipy>=1.6.0</li>
  <li>implicit>=0.4.2</li>
  <li>pyspark>=3.0.0</li>
  <li>tensorflow==1.14.0</li>
  <li>scikit-learn>=0.21.3</li>
  <li>xgboost>=1.1.1</li>
  <li>tffm==1.0.1</li>
  <li>pywFM==0.9.0</li>
</ul>
