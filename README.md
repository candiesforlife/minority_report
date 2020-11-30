# Data analysis
- Document here the project: minority_report
- Description: The minority report aims to predict the hourly intensity of crimes in a defined region in the next 48hours. (our y)
- Data Source: https://lewagon-alumni.slack.com/files/U01CE79R7K5/F01FLDM0KRC/nypd_complaint_historic_datadictionary.ods
- Type of analysis:

Please document the project the better you can.

# Startup the project

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
```


