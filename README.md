# Assignment 3

The third assignment consists of the construction of a predictive model using Spark (Structured) Streaming and textual data.

You will work with data coming from [Arxiv.org](https://arxiv.org/), the free distribution service and open-access archive for nearly 2.4 million scholarly articles in the fields of physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and systems science, and economics.

A data streamer server is already set up for you as follows:

- Published papers are monitored, and their categories, title, and abstract are extracted. We do so by polling the API every so often. Example endpoint: [https://export.arxiv.org/api/query?search_query=submittedDate:[202503251500+TO+202503261500]&max_results=2000](https://export.arxiv.org/api/query?search_query=submittedDate:[202503251500+TO+202503261500]&max_results=2000)

Next, the information is exposed through a streaming data source to you running at `seppe.net:7778`. When you connect to it, this will provide publications to you one by one:

- We fetch publications starting from now minus 96 hours ago and drip feed them over the connection
- We start from 96h ago so you can more easily test the connection by immediately receiving publications; the reason why we go so much back is because no publications are accepted during the weekend
- Next, whilst the stream is being kept open, we just continue to send newly published articles as-they-arrive

The stream is provided as a textual data source with one article per line, formatted as a JSON object, e.g.:

```
{
 "aid": "http://arxiv.org/abs/2503.19871v1", 
 "title": "A natural MSSM from a novel $\\mathsf{SO(10)}$ [...]", 
 "summary": "The $\\mathsf{SO(10)}$ model [...]", 
 "main_category": "hep-ph", 
 "categories": "hep-ph,hep-ex", 
 "published": "2025-03-25T17:36:54Z"
}
```

The goal of this assignment is threefold:

- 1 - Collect a historical set of data
    
    - _Important: get started with this as soon as possible. We will discuss Spark and text mining in more detail later on, but you can already start gathering your data_
- 2 - Construct a predictive model that predicts the categories an article belongs to:
    
    - There are different ways how you can approach this question: you can either try to predict the `main_category` (in which case it is a multiclass problem), or try to tackle it as a multilabel problem by trying to predict all of the `categories` (comma separated)
    - The second question is related to which categories you want to include: categories which contain a hyphen, such as `hep-ph` above, are a subcategory of the broader `hep`, so you might wish to reduce the number of classes by only focussing on the main categories (or focussing only on articles belong to a single main category, such as computer science, `cs`)
    - You can see all the categories over at [https://arxiv.org/](https://arxiv.org/)
    - You can use any predictive model you want to, but groups that incorporate a small LLM, or a more modern textual model ([https://huggingface.co/facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) is a very good start, for instance), will be rewarded for this
    - If you want to use a more traditional approach (TF-IDF plus a classifier, then try to use Spark's built-in ML models)
- 3 - Show that your model can make predictions in a "deployed" setting
    

**Setting up Spark**

Since the data set we'll work with is still relatively small, you will (luckily) not need a cluster of machines, but can run Spark locally on your machine (and save the data locally as well).

- First, download the ZIP file from [this link](https://seppe.net/aa/assignment3/spark.zip) and extract it somewhere, e.g. on your Desktop. This ZIP file contains the latest stable Spark version available at this time (3.5.5)
    
    - If you prefer to follow along with video instructions, an MP4 file is contained in the ZIP with a walkthrough for Windows and Mac users
- We will use `pixi.sh` to set up our environment. Pixi is a modern alternative to Conda, but you can use Conda as well (in which case you install the packages below using `conda install`). Download Pixi for your platform over at [https://github.com/prefix-dev/pixi/releases/](https://github.com/prefix-dev/pixi/releases/) and put the executable file (e.g. `pixi.exe` for Windows) in the `spark` folder you just extracted
    
- Next, we install all packages we need in the environment. In a Terminal or command line window, first navigate to the `spark`, and run: `pixi init` to initialize the environment. On Mac, use `./pixi` instead of `pixi` in these commands. Then run `pixi add python=3.11 pyspark findspark jupyter openjdk=11` to install the necessary packages
    
- Mac users will probably also have to make the Spark binaries executable (in case you get a PermissionError in the notebooks), you can do so by running this command in the `spark` directory `chmod +x ./spark-3.5.5-bin-hadoop3/bin*`
    
- You can then start Jupyter using `pixi run jupyter notebook`
    

**Example notebooks**

Once you have Jupyter open, explore the example notebooks under `notebooks`.

**Important: the first cell in these notebooks use `findspark` to initialize Spark and its contexts. You will need to add the same cell to all new notebooks you create.**

- `spark_example.ipynb`: Try this first! This is a simple Spark example to calculate pi and serves as check to see whether Spark is working correctly
- `spark_streaming_example.ipynb`: A simple Spark Streaming example that prints out the data you'll work with. This is a test to see whether you can receive the data
- `spark_streaming_example_saving.ipynb`: A simple Spark Streaming example that saves the data. Use this to get started saving your historical set
- `spark_streaming_example_predicting.ipynb`: A very naïve prediction approach
- `spark_structured_streaming_example.ipynb`: An example using Spark Structured Streaming

**Objective**

Using Spark, your task for this assignment is as follows:

- 1 - Collect a historical set of data
    - Get started with this as soon as possible
    - Make sure to set up Spark using the instructions posted above
- 2 - Construct a predictive model
    - The stream is text-based with each line containing one message (one instance) formatted as a JSON dictionary
    - You are strongly encouraged to build your model using `spark.ml` (MLlib), but you can use `scikit-learn` as a fallback
    - Alternatively, use a more modern model, as described above
    - Pick between the multiclass vs. multilabel, all categories vs. main categories tasks
- 3 - Show that your model can make predictions in a "deployed" setting
    - I.e. show that you can connect to the data source, preprocess/featurize incoming messages, have your model predict the label, and show it, similar to `spark_streaming_example_predicting.ipynb` (but using a smarter, real predictive model)
    - This means that you'll need to look for a way to save and load your trained model, if necessary
    - The goal is not to obtain a perfect predictive accuracy, but mainly to make sure you can set up Spark and work in a streaming environment

The third part of your lab report should contain:

- Overview of the steps above, the source code of your programs, as well as the output after running them
- Feel free to include screen shots or info on encountered challenges and how you dealt with them
- Even if your solution is not fully working or not working correctly, you can still receive marks for this assignment based on what you tried and how you'd need to improve your end result

**Further remarks**

- Get started with setting up Spark and fetching data as quickly as possible
- Make sure to have enough data to train your model. New publications arrive relatively slow (during some days, no articles might appear at all, whereas other days will be very busy)
- The data stream is line delimited with every line containing one instance in JSON format, but can be easily converted to a DataFrame (and RDD). The example notebooks give some ideas on how to do so
- You can use both Spark Streaming or Spark Structured Streaming
- Don't be afraid to ask e.g. ChatGPT or Claude for help to code up your approach, this is certainly permitted, but make sure not to get stuck in "vibe coding" where you have a notebook spanning twenty pages without knowing what you're really doing anymore
- Do let me know in case the streaming server would crash

_You do not hand in each assignment separately, but hand in your completed lab report containing all four assignments on Sunday June 1st. For an overview of the groups, see Toledo. Note for externals (i.e. anyone who will NOT partake in the exams -- this doesn't apply to normal students): you are free to partake in (any of) the assignments individually, but not required to._

**FAQ**

- The first cell in the example notebook fails (the one using `findspark`)
    
    - This cell attempts to do a couple of things: set the `SPARK_HOME` environment variable to the right directory; on Windows: set the `HADOOP_HOME` environment variable to the right `winutils` subfolder (necessary for Spark to work); initialize `findspark`, and then construct the different Spark contexts. Inspect the output and make sure the path names are correct
- Spark cannot be initialized, the notebook or command line shows an error "getSubject is supported only if a security manager is allowed"
    
    - Your `openjdk` version is too recent, make sure you have installed version 11 in your environment.
- Everything seems to work but I get a lot of warnings on the console or in the notebook
    
    - Spark is very verbose. On Mac, warnings are shown in the notebook itself, which makes it more annoying to use. If you don't like that, Google on how to stop Jupyter from capturing stderr
- I can't save the stream... everything else seems fine
    
    - Make sure you're calling the "saveAsTextfiles" function with "file:///" prepended to the path: `lines.saveAsTextFiles("file:///C:/...")`. Also make sure that the folder where you want to save the files exist. Note that the "saveAsTextfiles" method expects a _directory name_ as the argument. It will automatically create a folder for each mini-batch of data.
- Can I prevent the `saveAsTextFiles` function from creating so many directories and files?
    
    - You can first repartition the RDD to one partition before saving it: `lines.repartition(1).saveAsTextFiles("file:///C:/...")`. To prevent multiple directories, change the trigger time to e.g. `ssc = StreamingContext(sc, 60)`, though this will still create multiple directories. Setting the trigger interval higher is not really recommended, as you wouldn't want to lose data in case something goes wrong.
- So if I still end up with multiple directories, how do I read them in?
    
    - It's pretty easy to loop over subdirectories in Python. Alternatively, the `sc.textFile` command is pretty smart and can parse through multiple files in one go.
- Is it normal all my folders only contain `_SUCCESS` files but no actual data files?
    
    - That depends. A `_SUCCESS` file indicates that the mini-batch was saved correctly. `part-*` files contain the actual data. And files ending with `.crc` contain a checksum. It's normal if not all of your folders contain `part-*` data, when no data was received in that time frame. However, if none of your folders are having data, especially not when you have restarted the notebook, something else has gone wrong. Try the `spark_streaming_example.ipynb` notebook to verify whether you're at least receiving data at all.
- Is there a way how I can monitor Spark?
    
    - Yes, go to [http://127.0.0.1:4040/](http://127.0.0.1:4040/) in your browser while Spark is running and you'll get access to a monitoring dashboard. Under the "Environment" tab, you should be able to find a "spark.speculation" entry for instance w.r.t. the question above. Under "Jobs", "Stage", and "Streaming", you can get more info on how things are going.
- I'm trying to convert my saved files to a DataFrame, but Spark complains for some files?
    
    - Data is always messy, especially the ones provided by this instructor. Make sure you can handle badly formatted lines and discard them.
- My stream crashes after a while with an "RDD is empty" error...
    
    - Make sure you're checking for empty RDDs, e.g. `if rdd.isEmpty(): return`.
- I've managed to create a model. When I try to apply it on the stream, Spark crashes with a Hive / Derby error, e.g. when I try to .load() my model(s) or once the first RDD arrives
    
    - Check the example notebooks for ideas on how to load in your model in "globals()" once.
- When I call `ssc_t.stop()`, Spark never seems to stop the stream
    
    - You can try changing `stopGraceFully=True` to `False`. Even then, Spark might not want to stop its stream processing pipeline in case you're doing a lot with the incoming data, preventing Spark from cleaning up. Try decreasing the trigger time, or simply restart the Jupyter kernel to start over.
- Spark complains that only one StreamingContext can be active at a time (or "ValueError: Cannot run multiple SparkContexts at once")
    
    - A good idea is to (save and) close all running notebooks and start again fresh. Spark doesn't like having multiple contexts running, so it is best to only have one notebook running at a time. (Closing a tab with a notebook does not mean that the _kernel_ is stopped, however, check the "Running" tab on the Jupyter main page.)
- Why do I receive the same instances (or: why do I have instances twice) when reconnecting?
    
    - To make sure you are served data right away, the stream server starts from a while back and works its way back to the current time. You can remove duplicate instances based on the `aid` identifier.
- Can I use R?
    
    - There are two main Spark R packages available: `SparkR` (the official one) and `sparklyr` (from the folks at RStudio and fits better with the tidyverse). You can try using these, but you'll have to do some setting up in order so R can find your Spark installation. I'd strongly recommend using Python.
- The server is just a socket server, so can't we just get the data that way?
    
    - For those who know, yes, basically: `nc seppe.net 7778`, so indeed in this case it would be easy to do this in Python directly.