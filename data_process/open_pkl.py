import pickle as p 

#filepath = 'sub04_meg_story22.pkl'
filepath = 'meg_vector/word_meg_story22.pkl'
F=open(filepath,'rb')

content=p.load(F)
print(content)
#print(content.shape)
print(len(content))