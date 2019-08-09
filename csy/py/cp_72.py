fnames = 'pensees_30m_sq8_idmap_29m.txt'
f2 = 'pensees_30m_sq8_idmap_72.txt'
with open(fnames,'a+') as f1:
    i = 0
    for line in open(f2,'r'):
        line=line.strip()
        i += 1
        num = "%03d%06d\n" % (999,i)
        l = line +' '+ num
        #print(l)
        f1.write(l)
