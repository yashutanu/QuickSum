import os
from graphviz import Source
from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree ,ParentedTree

os.environ['STANFORD_PARSER'] = '/home/yashu/8TH_PRO/Final/jars'
os.environ['STANFORD_MODELS'] = '/home/yashu/8TH_PRO/sentence_lexical_sub/jars'

english_parser = StanfordParser("/home/yashu/8TH_PRO/Final/jars/stanford-parser.jar",'/home/yashu/8TH_PRO/Final/jars/stanford-parser-3.5.2-models.jar')

def traverse(t):
   global result_store
   if t.label()== ".":
        return
   for child in t:
        if type(child) == Tree and child.label() !='PRN' and child.label()!='NP-TMP' and child.label() !='ADVP' and child.label()!='PP':
            traverse(child)
        elif type(child)== str:    
            result_store+=(" "+t[0])
   return result_store

result_store=""
"""
lt=list(english_parser.raw_parse("The central and provincial governments have invested 160 million yuan (nearly 20 million US dollars) into panda protection programs since 1992."))

print("sentence: The central and provincial governments have invested 160 million yuan (nearly 20 million US dollars) into panda protection programs since 1992.") 
r=traverse(lt[0])
print("simplified sentence: ",r)


for line in lt:
    for sentence in line:
        sentence.draw()
"""
