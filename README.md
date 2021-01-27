# Minority Report

Minority Report aims to predict future crime intensity in the 75th precinct in NYC.

Each crime occurence is represented in a 3-D tensor made up of latitude, longitude and an indicated timeframe.

After passing the tensors through a Gaussian filter and stacking them, they are used to train a Convolutional Neural Network.

You can find out more by checking out our presentation [here](https://docs.google.com/presentation/d/1LlsR1xTr1Hx4iTx-in5fL0dMzcIrm319WJlsk2z9HWU/edit?usp=sharing)

[NYPD Data Source](https://catalog.data.gov/dataset/nypd-complaint-data-historic/resource/427e1d35-8a14-4e6b-b7ce-f5c45fb30b26)

# How to run Minority Report

In Terminal:

1. python minority_report/clean_data.py

2. python minority_report/clean_split.py

3. python minority_report/trainer.py

N.B. Model has to be initiated in google collab.

# Minority Report for Viz Purposes

In Terminal:

1. python minority_report/clean_data.py

2. python minority_report/viz.py

<!-- TO DO:
- Rename matrix.py to preprocessing.py
- Have py file that runs full cleaning and preprocessing in one
- Have trainer run smoothly as python file with model -->

<!-- # Data analysis
- Document here the project: minority_report
- Description: The minority report aims to predict the hourly intensity of crimes in a defined region in the next 48hours. (our y)
- Data Source: https://catalog.data.gov/dataset/nypd-complaint-data-historic/resource/427e1d35-8a14-4e6b-b7ce-f5c45fb30b26
- Type of analysis:

Please document the project the better you can. -->

<!-- # Startup the project

The initial setup.


Create virtualenv and install the project:
```bash
  $ sudo apt-get install virtualenv python-pip python-dev
  $ deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
  $ make clean install test
```

Check for minority_report in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/minority_report`
- Then populate it:

```bash
  $ ##   e.g. if group is "{group}" and project_name is "minority_report"
  $ git remote add origin git@gitlab.com:{group}/minority_report.git
  $ git push -u origin master
  $ git push -u origin --tags
```

Functionnal test with a script:
```bash
  $ cd /tmp
  $ minority_report-run
```
# Install
Go to `gitlab.com/{group}/minority_report` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:
```bash
  $ sudo apt-get install virtualenv python-pip python-dev
  $ deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:
```bash
  $ git clone gitlab.com/{group}/minority_report
  $ cd minority_report
  $ pip install -r requirements.txt
  $ make clean install test                # install and test
```
Functionnal test with a script:
```bash
  $ cd /tmp
  $ minority_report-run
``` -->


