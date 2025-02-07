# AppliedStatistics

## Name conventions

* Use the prefix car_xxx and house_xxx for the respective files. 
* Use a number to indicate the steps (0 = import data, 1= data cleansing, 2= transformation, 3=visualizations, etc. see below), let's see if we can follow this that much
* Use a short description, of what the file is about 
Example: car_03_EDA.ipynb

## Manage repository

To have aa better overview of who does what, create "personal branches"
Create a local new branch from the master branch for different tasks, e.g. "car_EDA_vanessa":
+ `git checkout -b car_EDA_vanessa`

Make sure your new branch is rebased to master branch:
+ `git rebase master`

If master has been updated, rebase the local branch:
+ `git checkout master`
+ `git pull origin master`
+ `git checkout car_EDA_vanessa`
+ `git rebase master`

After finishing the part, commit and push:

+ `git add . `
+ `git commit -"some commit message"`
+ `git push origin --force car_EDA_vanessa`

When a local branch is ready, merge into master branch, if necessary, solve conflicts.

## Workflow
Let's follow the data science mining process:

<img src="images/programming-process.png">

1. Import data First you must import your data into Python. This typically means that you take
data stored in a file, database, or web API, and load it into a DataFrame in Python (using pandas).

2. Tidying data Once you’ve imported your data, it is a good idea to tidy it. Tidying your data
means storing it in a consistent form that matches the semantics of the dataset with the way it is
stored. In brief, when your data is tidy, each column is a variable, and each row is an observation.
Tidy data is important because the consistent structure lets you focus your struggle on questions
about the data.

3. Transform data Once you have tidy data, a common first step is to transform it. Transformation
includes narrowing in on observations of interest (like all people in one city, or all data
from the last year), creating new variables that are functions of existing variables (like computing
velocity from speed and time), and calculating a set of summary statistics (like counts or means).
Together, tidying and transforming are called wrangling, because getting your data in a form
that’s natural to work with often feels like a fight!

4. Visualize data Once you have tidy data with the variables you need, there are two main
engines of knowledge generation: visualisation and modelling. These have complementary
strengths and weaknesses so any real analysis will iterate between them many times.
Visualisation is a fundamentally human activity. A good visualisation will show you things
that you did not expect, or raise new questions about the data. A good visualisation might also
hint that you’re asking the wrong question, or you need to collect different data. Visualisations
can surprise you, but don’t scale particularly well because they require a human to interpret them.


5. Models Models are complementary tools to visualisation. Once you have made your questions
sufficiently precise, you can use a model to answer them. Models are a fundamentally mathematical
or computational tool, so they generally scale well. But every model makes assumptions,
and by its very nature a model cannot question its own assumptions. That means a model cannot
fundamentally surprise you.

6. Communication The last step is communication, an absolutely critical part of any data analysis
project. It doesn’t matter how well your models and visualisation have led you to understand
the data unless you can also communicate your results to others.
Surrounding all these tools is programming. Programming is a cross-cutting tool that you use
in every part of the project. You don’t need to be an expert programmer to be a data scientist, but
learning more about programming pays off because becoming a better programmer allows you to
automate common tasks, and solve new problems with greater ease.
Source: Wickham/Grolemund, 2016