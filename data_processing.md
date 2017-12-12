
## 2016 Election Project 
### Part 1 of Processing Pipeline

This notebook is intended to document my data processing throughout this project. I'll be poking around and modifying my data in this file. The data I am starting out with are transcripts of the presidential debates from the 2016 US Election. I am processing the Democratic and Republican primary debates, and the debates of the general election between Hillary Clinton and Donald Trump. The transcripts were taken from UCSB's American Presidency Project, and the citation for each of the transcripts can be found in the README.

### Table of Contents
- [Reading in Transcript Files](#reading-in-files)
- [Splitting Transcripts by Speaker](#splitting-transcripts-by-speaker)
- [Tokenizing Each Speaker's Sentences](#tokenizing-each-speakers-sentences)
- [Mapping to Debate Type](#)
- [Reordering and Naming Columns](#reordering-and-naming-columns)
- [Saving DataFrames](#saving-dataframes)


```python
%pprint
import nltk
from nltk.corpus import PlaintextCorpusReader
import pandas as pd
import glob
import os
import re
```

    Pretty printing has been turned OFF


### Reading In Files


```python
os.chdir('/Users/Paige/Documents/Data_Science/2016-Election-Project/data/Debates/transcripts/')
files = glob.glob("*.txt")
files
```




    ['1-14-16_rep.txt', '1-17-16_dem.txt', '1-25-16_dem.txt', '1-28-16_rep.txt', '10-13-15_dem.txt', '10-19-16.txt', '10-28-15_rep.txt', '10-9-16.txt', '11-10-15_rep.txt', '11-14-15_dem.txt', '12-15-15_rep.txt', '12-19-15_dem.txt', '2-11-16_dem.txt', '2-13-16_rep.txt', '2-25-16_rep.txt', '2-4-16_dem.txt', '2-6-16_rep.txt', '3-10-16_rep.txt', '3-3-16_rep.txt', '3-6-16_dem.txt', '3-9-16_dem.txt', '4-14-16_dem.txt', '8-6-15_rep.txt', '9-16-15_rep.txt', '9-26-16.txt']




```python
len(files)
```




    25




```python
#I'm creating a list where each entry in the list is a transcript
transcripts = []
for f in files:
    fi = open(f, 'r')
    txt = fi.read()
    fi.close
    transcripts.append(txt)
```


```python
print(transcripts[0][:200])
```

    PARTICIPANTS:
    Former Governor Jeb Bush (FL);
    Ben Carson;
    Governor Chris Christie (NJ);
    Senator Ted Cruz (TX);
    Governor John Kasich (OH);
    Senator Marco Rubio (FL);
    Donald Trump;
    MODERATORS:
    Maria Barti



```python
words0= nltk.word_tokenize(transcripts[0])
len(words0)
```




    27658




```python
sents0= nltk.sent_tokenize(transcripts[0])
len(sents0)
```




    1498




```python
print(transcripts[1][:200])
```

    PARTICIPANTS:
    Former Secretary of State Hillary Clinton;
    Former Governor Martin O'Malley (MD);
    Senator Bernie Sanders (VT);
    MODERATORS:
    Lester Holt (NBC News)
    Andrea Mitchell (NBC News)
    
    HOLT: Good ev


**Cleaning up: I would eventually like to end up with a dataframe where the columns are Date, Type (primary or general), Speaker, Sents, where the Sents are in the order that they are said.**

### Splitting Transcripts by Speaker


```python
#I want to split large chunks of the transcript based on who is speaking.
#Since the transcript data has a pretty standardized fomat (The speaker is in all caps followed by a colon)
#I can add a marker to each of these sections, and split the data on that marker

speaker_split = []

for txt in transcripts:
    #To take care of the first one where there is no newline preceding the label..
    txt = txt.replace("PARTICIPANTS:", 'PARTICIPANTS%:')
    #get rid of [through translator] label in 3-19-16 debate
    txt = txt.replace(" [through translator]:", ":")
    #The ' in Martin O'Malley's name was causing some issues so I'm changing his name (for the speaker column)
    #to OMALLEY
    txt = txt.replace("O'MALLEY:", "OMALLEY:")
    txt = re.sub(r"\n([A-Z]+)(\s[A-Z]+)?:", r"#$&\1%:", txt)
    speaker_split.append(txt)

#Split each chunk by the special marker
speaker_split = [txt.strip().split("#$&") for txt in speaker_split]
```


```python
speaker_split[0][:4]
```




    ['PARTICIPANTS%:\nFormer Governor Jeb Bush (FL);\nBen Carson;\nGovernor Chris Christie (NJ);\nSenator Ted Cruz (TX);\nGovernor John Kasich (OH);\nSenator Marco Rubio (FL);\nDonald Trump;', 'MODERATORS%:\nMaria Bartiromo (Fox Business Network); and\nNeil Cavuto (Fox Business Network)\n', "CAVUTO%: It is 9:00 p.m. here at the North Charleston Coliseum and Performing Arts Center in South Carolina. Welcome to the sixth Republican presidential of the 2016 campaign, here on the Fox Business Network. I'm Neil Cavuto, alongside my friend and co-moderator Maria Bartiromo.\n", 'BARTIROMO%: Tonight we are working with Facebook to ask the candidates the questions voters want answered. And according to Facebook, the U.S. election has dominated the global conversation, with 131 million people talking about the 2016 race. That makes it the number one issue talked about on Facebook last year worldwide.\n']




```python
#Creating a giant list so I don't have to handle things one at a time
#Splitting each chunk into two elements: speaker, speech
debates = [[txt.split("%:") for txt in split] for split in speaker_split]
debates[0][:4]
```




    [['PARTICIPANTS', '\nFormer Governor Jeb Bush (FL);\nBen Carson;\nGovernor Chris Christie (NJ);\nSenator Ted Cruz (TX);\nGovernor John Kasich (OH);\nSenator Marco Rubio (FL);\nDonald Trump;'], ['MODERATORS', '\nMaria Bartiromo (Fox Business Network); and\nNeil Cavuto (Fox Business Network)\n'], ['CAVUTO', " It is 9:00 p.m. here at the North Charleston Coliseum and Performing Arts Center in South Carolina. Welcome to the sixth Republican presidential of the 2016 campaign, here on the Fox Business Network. I'm Neil Cavuto, alongside my friend and co-moderator Maria Bartiromo.\n"], ['BARTIROMO', ' Tonight we are working with Facebook to ask the candidates the questions voters want answered. And according to Facebook, the U.S. election has dominated the global conversation, with 131 million people talking about the 2016 race. That makes it the number one issue talked about on Facebook last year worldwide.\n']]



### Tokenizing Each Speaker's Sentences


```python
debate_sents = []
#For each debate, then for each [speaker, speech] chunk in that debate, get a list of tokenized sents to replace the speech
for debate in debates:
    sents_toks = []
    for chunk in debate:
        sents = nltk.sent_tokenize(chunk[1])
        for sent in sents:
            sents_toks.append([chunk[0], sent])
    debate_sents.append(sents_toks)
```

### Mapping to Debate Type


```python
#I am creating a list of 25 dataframes, one for each debate
# Adding a column specifying the type of debate, the date, the speaker, and sent

dataframes = []
for f in files:
    index = files.index(f)
    df = pd.DataFrame(debate_sents[index])
    if f.endswith('_dem.txt'):
        df['Type'] = 'primary_dem' 
        df['Date'] = f[:-8]
    elif f.endswith('_rep.txt'):
        df['Type'] = 'primary_rep' 
        df['Date'] = f[:-8]
    else:
        df['Type'] = 'general' 
        df['Date'] = f[:-4]
    dataframes.append(df)
```


```python
# Every returned Out[] is displayed, not just the last one. 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


```python
for df in dataframes:
    df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Governor Jeb Bush (FL);\nBen Carson;\...</td>
      <td>primary_rep</td>
      <td>1-14-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nMaria Bartiromo (Fox Business Network); and\...</td>
      <td>primary_rep</td>
      <td>1-14-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CAVUTO</td>
      <td>It is 9:00 p.m. here at the North Charleston ...</td>
      <td>primary_rep</td>
      <td>1-14-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CAVUTO</td>
      <td>Welcome to the sixth Republican presidential o...</td>
      <td>primary_rep</td>
      <td>1-14-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CAVUTO</td>
      <td>I'm Neil Cavuto, alongside my friend and co-mo...</td>
      <td>primary_rep</td>
      <td>1-14-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton;\n...</td>
      <td>primary_dem</td>
      <td>1-17-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nLester Holt (NBC News)\nAndrea Mitchell (NBC...</td>
      <td>primary_dem</td>
      <td>1-17-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HOLT</td>
      <td>Good evening and welcome to the NBC News Yout...</td>
      <td>primary_dem</td>
      <td>1-17-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HOLT</td>
      <td>After all the campaigning, soon, Americans wil...</td>
      <td>primary_dem</td>
      <td>1-17-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HOLT</td>
      <td>And New Hampshire not far behind.</td>
      <td>primary_dem</td>
      <td>1-17-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton;\n...</td>
      <td>primary_dem</td>
      <td>1-25-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATOR</td>
      <td>\nChris Cuomo, CNN</td>
      <td>primary_dem</td>
      <td>1-25-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUOMO</td>
      <td>All right.</td>
      <td>primary_dem</td>
      <td>1-25-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUOMO</td>
      <td>We are live at Drake University in Des Moines,...</td>
      <td>primary_dem</td>
      <td>1-25-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUOMO</td>
      <td>Welcome to our viewers in the United States an...</td>
      <td>primary_dem</td>
      <td>1-25-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Governor Jeb Bush (FL);\nBen Carson;\...</td>
      <td>primary_rep</td>
      <td>1-28-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nBret Baier (Fox News);\nMegyn Kelly (Fox New...</td>
      <td>primary_rep</td>
      <td>1-28-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BAIER</td>
      <td>Nine p.m. on the East Coast.</td>
      <td>primary_rep</td>
      <td>1-28-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BAIER</td>
      <td>Eight o'clock here in Des Moines, Iowa.</td>
      <td>primary_rep</td>
      <td>1-28-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BAIER</td>
      <td>Welcome to the seventh Republican presidential...</td>
      <td>primary_rep</td>
      <td>1-28-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Governor Lincoln Chafee (RI);\nFormer...</td>
      <td>primary_dem</td>
      <td>10-13-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nAnderson Cooper (CNN);\nDana Bash (CNN);\nDo...</td>
      <td>primary_dem</td>
      <td>10-13-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COOPER</td>
      <td>I'm Anderson Cooper.</td>
      <td>primary_dem</td>
      <td>10-13-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>COOPER</td>
      <td>Thanks for joining us.</td>
      <td>primary_dem</td>
      <td>10-13-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>COOPER</td>
      <td>We've already welcomed the candidates on stage.</td>
      <td>primary_dem</td>
      <td>10-13-15</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton (D...</td>
      <td>general</td>
      <td>10-19-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATOR</td>
      <td>\nChris Wallace (Fox News)</td>
      <td>general</td>
      <td>10-19-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WALLACE</td>
      <td>Good evening from the Thomas and Mack Center ...</td>
      <td>general</td>
      <td>10-19-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WALLACE</td>
      <td>I'm Chris Wallace of Fox News, and I welcome y...</td>
      <td>general</td>
      <td>10-19-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WALLACE</td>
      <td>This debate is sponsored by the Commission on ...</td>
      <td>general</td>
      <td>10-19-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Governor Jeb Bush (FL);\nBen Carson;\...</td>
      <td>primary_rep</td>
      <td>10-28-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nJohn Harwood (CNBC);\nBecky Quick (CNBC); an...</td>
      <td>primary_rep</td>
      <td>10-28-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>QUINTANILLA</td>
      <td>Good evening, I'm Carl Quintanilla, with my c...</td>
      <td>primary_rep</td>
      <td>10-28-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>QUINTANILLA</td>
      <td>We'll be joined tonight by some of CNBC's top ...</td>
      <td>primary_rep</td>
      <td>10-28-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>QUINTANILLA</td>
      <td>Let's get through the rules of the road.</td>
      <td>primary_rep</td>
      <td>10-28-15</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton (D...</td>
      <td>general</td>
      <td>10-9-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nAnderson Cooper (CNN) and\nMartha Raddatz (A...</td>
      <td>general</td>
      <td>10-9-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RADDATZ</td>
      <td>Ladies and gentlemen the Republican nominee f...</td>
      <td>general</td>
      <td>10-9-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RADDATZ</td>
      <td>[applause]</td>
      <td>general</td>
      <td>10-9-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>COOPER</td>
      <td>Thank you very much for being here.</td>
      <td>general</td>
      <td>10-9-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Governor Jeb Bush (FL);\nBen Carson;\...</td>
      <td>primary_rep</td>
      <td>11-10-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nGerard Baker (The Wall Street Journal);\nMar...</td>
      <td>primary_rep</td>
      <td>11-10-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CAVUTO</td>
      <td>It is 9:00 p.m. on the East Coast, 8:00 p.m. ...</td>
      <td>primary_rep</td>
      <td>11-10-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CAVUTO</td>
      <td>Welcome to the Republican presidential debate ...</td>
      <td>primary_rep</td>
      <td>11-10-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CAVUTO</td>
      <td>I'm Neil Cavuto, alongside my co-moderators, M...</td>
      <td>primary_rep</td>
      <td>11-10-15</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton;\n...</td>
      <td>primary_dem</td>
      <td>11-14-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nNancy Cordes (CBS News);\nKevin Cooney (CBS ...</td>
      <td>primary_dem</td>
      <td>11-14-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DICKERSON</td>
      <td>Before we start the debate here are the rules.</td>
      <td>primary_dem</td>
      <td>11-14-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DICKERSON</td>
      <td>The candidates have one minute to respond to o...</td>
      <td>primary_dem</td>
      <td>11-14-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DICKERSON</td>
      <td>Any candidate who is attacked by another candi...</td>
      <td>primary_dem</td>
      <td>11-14-15</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Governor Jeb Bush (FL);\nBen Carson;\...</td>
      <td>primary_rep</td>
      <td>12-15-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nWolf Blitzer (CNN);\nDana Bash (CNN); and\nH...</td>
      <td>primary_rep</td>
      <td>12-15-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BLITZER</td>
      <td>Welcome to the CNN-Facebook Republican presid...</td>
      <td>primary_rep</td>
      <td>12-15-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BLITZER</td>
      <td>We have a very enthusiastic audience.</td>
      <td>primary_rep</td>
      <td>12-15-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BLITZER</td>
      <td>Everyone is here.</td>
      <td>primary_rep</td>
      <td>12-15-15</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton;\n...</td>
      <td>primary_dem</td>
      <td>12-19-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nMartha Raddatz (ABC News)\nDavid Muir (ABC N...</td>
      <td>primary_dem</td>
      <td>12-19-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RADDATZ</td>
      <td>Good evening to you all.</td>
      <td>primary_dem</td>
      <td>12-19-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RADDATZ</td>
      <td>The rules for tonight are very basic and have ...</td>
      <td>primary_dem</td>
      <td>12-19-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RADDATZ</td>
      <td>Candidates can take up to a minute-and-a-half ...</td>
      <td>primary_dem</td>
      <td>12-19-15</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton;\n...</td>
      <td>primary_dem</td>
      <td>2-11-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nGwen Ifill (PBS);\nJudy Woodruff (PBS)</td>
      <td>primary_dem</td>
      <td>2-11-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WOODRUFF</td>
      <td>Good evening, and thank you.</td>
      <td>primary_dem</td>
      <td>2-11-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WOODRUFF</td>
      <td>We are happy to welcome you to Milwaukee for t...</td>
      <td>primary_dem</td>
      <td>2-11-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WOODRUFF</td>
      <td>We are especially pleased to thank our partner...</td>
      <td>primary_dem</td>
      <td>2-11-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Governor Jeb Bush (FL);\nBen Carson;\...</td>
      <td>primary_rep</td>
      <td>2-13-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATOR</td>
      <td>\nJohn Dickerson (CBS News); with</td>
      <td>primary_rep</td>
      <td>2-13-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PANELISTS</td>
      <td>\nMajor Garrett (CBS News); and\nKimberly Stra...</td>
      <td>primary_rep</td>
      <td>2-13-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DICKERSON</td>
      <td>Good evening.</td>
      <td>primary_rep</td>
      <td>2-13-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DICKERSON</td>
      <td>I'm John Dickerson.</td>
      <td>primary_rep</td>
      <td>2-13-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nBen Carson;\nSenator Ted Cruz (TX);\nGoverno...</td>
      <td>primary_rep</td>
      <td>2-25-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATOR</td>
      <td>\nWolf Blitzer (CNN); with</td>
      <td>primary_rep</td>
      <td>2-25-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PANELISTS</td>
      <td>\nMaria Celeste Arrarás (Telemundo);\nDana Bas...</td>
      <td>primary_rep</td>
      <td>2-25-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BLITZER</td>
      <td>We're live here at the University of Houston ...</td>
      <td>primary_rep</td>
      <td>2-25-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BLITZER</td>
      <td>[applause]\n\nAn enthusiastic crowd is on hand...</td>
      <td>primary_rep</td>
      <td>2-25-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton;\n...</td>
      <td>primary_dem</td>
      <td>2-4-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nChuck Todd (MSNBC);\nRachel Maddow (MSNBC)</td>
      <td>primary_dem</td>
      <td>2-4-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TODD</td>
      <td>Good evening, and welcome to the MSNBC Democr...</td>
      <td>primary_dem</td>
      <td>2-4-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MADDOW</td>
      <td>We are super excited to be here at the Univer...</td>
      <td>primary_dem</td>
      <td>2-4-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MADDOW</td>
      <td>Tonight, this is the first time that Hillary C...</td>
      <td>primary_dem</td>
      <td>2-4-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Governor Jeb Bush (FL);\nBen Carson;\...</td>
      <td>primary_rep</td>
      <td>2-6-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nDavid Muir (ABC News); and\nMartha Raddatz (...</td>
      <td>primary_rep</td>
      <td>2-6-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MUIR</td>
      <td>Good evening, again, everyone.</td>
      <td>primary_rep</td>
      <td>2-6-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MUIR</td>
      <td>This is the first time since Iowa and the only...</td>
      <td>primary_rep</td>
      <td>2-6-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MUIR</td>
      <td>The people of Iowa have been heard.</td>
      <td>primary_rep</td>
      <td>2-6-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nSenator Ted Cruz (TX);\nGovernor John Kasich...</td>
      <td>primary_rep</td>
      <td>3-10-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nJake Tapper (CNN);\nDana Bash (CNN);\nHugh H...</td>
      <td>primary_rep</td>
      <td>3-10-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TAPPER</td>
      <td>Live from the Bank United Center on the campu...</td>
      <td>primary_rep</td>
      <td>3-10-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TAPPER</td>
      <td>For our viewers in the United States and aroun...</td>
      <td>primary_rep</td>
      <td>3-10-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TAPPER</td>
      <td>In just five days voters will go to the polls ...</td>
      <td>primary_rep</td>
      <td>3-10-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nSenator Ted Cruz (TX);\nGovernor John Kasich...</td>
      <td>primary_rep</td>
      <td>3-3-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nBret Baier (Fox News);\nMegyn Kelly (Fox New...</td>
      <td>primary_rep</td>
      <td>3-3-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KELLY</td>
      <td>Good evening, and welcome to the fabulous FOX...</td>
      <td>primary_rep</td>
      <td>3-3-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KELLY</td>
      <td>I'm Megyn Kelly, along with my co-moderators, ...</td>
      <td>primary_rep</td>
      <td>3-3-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BAIER</td>
      <td>59 Republican delegates are at stake here in ...</td>
      <td>primary_rep</td>
      <td>3-3-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton;\n...</td>
      <td>primary_dem</td>
      <td>3-6-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nAnderson Cooper (CNN);\nDon Lemon (CNN)</td>
      <td>primary_dem</td>
      <td>3-6-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COOPER</td>
      <td>And welcome to The Whiting Auditorium on the ...</td>
      <td>primary_dem</td>
      <td>3-6-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>COOPER</td>
      <td>I'm Anderson Cooper.</td>
      <td>primary_dem</td>
      <td>3-6-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>COOPER</td>
      <td>I want to welcome our viewers in the United St...</td>
      <td>primary_dem</td>
      <td>3-6-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton;\n...</td>
      <td>primary_dem</td>
      <td>3-9-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nJorge Ramos (Univision);\nMaría Elena Salina...</td>
      <td>primary_dem</td>
      <td>3-9-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RAMOS</td>
      <td>[Speaking in Spanish]</td>
      <td>primary_dem</td>
      <td>3-9-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SALINAS</td>
      <td>This will be the first and only debate the ca...</td>
      <td>primary_dem</td>
      <td>3-9-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RAMOS</td>
      <td>Here with us tonight is Karen Tumulty, Washin...</td>
      <td>primary_dem</td>
      <td>3-9-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton;\n...</td>
      <td>primary_dem</td>
      <td>4-14-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATOR</td>
      <td>\nWolf Blitzer (CNN);</td>
      <td>primary_dem</td>
      <td>4-14-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PANELISTS</td>
      <td>\nDana Bash (CNN); and\nErrol Louis (NY1)</td>
      <td>primary_dem</td>
      <td>4-14-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BLITZER</td>
      <td>Secretary Clinton and Senator Sanders, you ca...</td>
      <td>primary_dem</td>
      <td>4-14-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BLITZER</td>
      <td>As moderator, I'll guide the discussion, askin...</td>
      <td>primary_dem</td>
      <td>4-14-16</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Governor Jeb Bush (FL);\nBen Carson;\...</td>
      <td>primary_rep</td>
      <td>8-6-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nBret Baier (Fox News);\nMegyn Kelly (Fox New...</td>
      <td>primary_rep</td>
      <td>8-6-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KELLY</td>
      <td>Welcome to the first debate night of the 2016...</td>
      <td>primary_rep</td>
      <td>8-6-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KELLY</td>
      <td>I'm Megyn Kelly... [applause]... along with my...</td>
      <td>primary_rep</td>
      <td>8-6-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KELLY</td>
      <td>Tonight... [applause] Nice.</td>
      <td>primary_rep</td>
      <td>8-6-15</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Governor Jeb Bush (FL);\nBen Carson;\...</td>
      <td>primary_rep</td>
      <td>9-16-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATORS</td>
      <td>\nJake Tapper (CNN);\nDana Bash (CNN); and\nHu...</td>
      <td>primary_rep</td>
      <td>9-16-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TAPPER</td>
      <td>I'm Jake Tapper.</td>
      <td>primary_rep</td>
      <td>9-16-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TAPPER</td>
      <td>We're live at the Ronald Reagan Library in Sim...</td>
      <td>primary_rep</td>
      <td>9-16-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TAPPER</td>
      <td>Round 2 of CNN's presidential debate starts now.</td>
      <td>primary_rep</td>
      <td>9-16-15</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Type</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PARTICIPANTS</td>
      <td>\nFormer Secretary of State Hillary Clinton (D...</td>
      <td>general</td>
      <td>9-26-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MODERATOR</td>
      <td>\nLester Holt (NBC News)</td>
      <td>general</td>
      <td>9-26-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HOLT</td>
      <td>Good evening from Hofstra University in Hemps...</td>
      <td>general</td>
      <td>9-26-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HOLT</td>
      <td>I'm Lester Holt, anchor of "NBC Nightly News."</td>
      <td>general</td>
      <td>9-26-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HOLT</td>
      <td>I want to welcome you to the first presidentia...</td>
      <td>general</td>
      <td>9-26-16</td>
    </tr>
  </tbody>
</table>
</div>



### Reordering and Naming Columns


```python
#Creating a new giant list of cleaned dataframes where the columns are reordered and cleaned up
dataframes_clean = []
for df in dataframes:
    #Drop the first two rows because they don't matter
    df.drop(0, inplace=True)
    df.drop(1, inplace=True)
    #Renaming the first two columns
    df.columns = ['Speaker', 'Sents', 'Debate Type', 'Date']
    #Strip newlines from Speaker and Sents columns
    df['Speaker'] = df['Speaker'].apply(lambda x: x.strip('\n'))
    df['Sents'] = df['Sents'].apply(lambda x: x.strip('\n'))
    #Reorder columns
    dataframes_clean.append(df[['Date','Debate Type', 'Speaker', 'Sents']])
```


```python
dataframes_clean[0].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Debate Type</th>
      <th>Speaker</th>
      <th>Sents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1-14-16</td>
      <td>primary_rep</td>
      <td>CAVUTO</td>
      <td>It is 9:00 p.m. here at the North Charleston ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1-14-16</td>
      <td>primary_rep</td>
      <td>CAVUTO</td>
      <td>Welcome to the sixth Republican presidential o...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1-14-16</td>
      <td>primary_rep</td>
      <td>CAVUTO</td>
      <td>I'm Neil Cavuto, alongside my friend and co-mo...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1-14-16</td>
      <td>primary_rep</td>
      <td>BARTIROMO</td>
      <td>Tonight we are working with Facebook to ask t...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1-14-16</td>
      <td>primary_rep</td>
      <td>BARTIROMO</td>
      <td>And according to Facebook, the U.S. election h...</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataframes_clean[-1].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Debate Type</th>
      <th>Speaker</th>
      <th>Sents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>9-26-16</td>
      <td>general</td>
      <td>HOLT</td>
      <td>Good evening from Hofstra University in Hemps...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9-26-16</td>
      <td>general</td>
      <td>HOLT</td>
      <td>I'm Lester Holt, anchor of "NBC Nightly News."</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9-26-16</td>
      <td>general</td>
      <td>HOLT</td>
      <td>I want to welcome you to the first presidentia...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9-26-16</td>
      <td>general</td>
      <td>HOLT</td>
      <td>The participants tonight are Donald Trump and ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9-26-16</td>
      <td>general</td>
      <td>HOLT</td>
      <td>This debate is sponsored by the Commission on ...</td>
    </tr>
  </tbody>
</table>
</div>



**Now I have a nice data frame for each debate. For any utterance in any debate, I provide information about who said it, what kind of debate it was, and when the debate took place. Now I'm going to export these dataframes to CSV files and process them with NER annotation in a different notebook.**

### Saving DataFrames


```python
#i=-1
#for df in dataframes_clean:
#    i+=1
#    df.to_csv('../csv/'+str(files[i][:-4])+'.csv')
```


```python
import pickle
f = open('/Users/Paige/Documents/Data_Science/dataframes_list.p', 'wb')
pickle.dump(dataframes_clean, f, -1)
f.close()
```
