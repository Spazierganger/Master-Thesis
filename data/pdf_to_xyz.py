s = """
5
HCl + O3, vdW complex , FC−CCSD/maug−cc−PVTZ
O 0. 1 2 3 0 2 6 1. 5 7 9 9 9 0 0. 0 7 1 7 6 8
H 0. 0 0 0 3 9 9 −1.480663 −0.251177
Cl −0.173959 −2.741270 −0.134041
O −0.428080 1. 1 5 1 3 4 8 1. 1 0 3 4 2 6
O 0. 4 7 2 7 7 1 0. 7 4 1 8 2 4 −0.789977
5
HCl + O3, t r a n s i t i o n s t r u c t u r e , FC−CCSD/maug−cc−PVTZ
O −0.273647 0. 9 7 4 7 4 8 −0.048949
H 0. 0 4 8 3 7 2 −0.746878 0. 5 8 1 0 9 5
Cl −0.178663 −1.801590 −0.600961
O 0. 0 3 4 7 0 3 0. 3 1 0 0 7 1 1. 0 7 6 4 6 6
O 0. 3 6 3 3 9 4 0. 5 1 4 8 7 6 −1.007648
5
HCl + O3, product , FC−CCSD/maug−cc−PVTZ
O 0. 5 2 3 2 6 7 0. 5 5 4 1 3 8 0. 2 8 5 8 5 7
H −0.860642 −0.368711 1. 1 2 8 7 5 0
Cl 0. 6 0 0 6 5 1 −1.578928 −1.095194
O −0.817830 0. 5 0 7 8 7 8 0. 7 2 3 4 5 0
O 0. 5 4 8 7 1 3 0. 1 3 6 8 5 2 −1.042863
""".replace('−', '-')

last_was_one = False
for l in s.split('\n'):
    split = l.split(' ')
    if len(split) < 2:
        last_was_one = True
        if len(l) > 0:
            print(l)
        continue
    else:
        if last_was_one:
            last_was_one = False
            if 'complex' in l:
                print('complex')
                continue
            if 't r a n s i t i o n s t r u c t u r e' in l:
                print('transition')
                continue
            elif 'p r o d u c t' in l:
                print('product')
                continue
        else:
            to_print = ''
            counter = -1
            for s in split:
                if s.isalpha():
                    to_print += s + ' '
                elif s.startswith('−'):
                    to_print += s + ' '
                elif s.isnumeric():
                    to_print += s
                    if counter >= 0:
                        counter += 1
                    if counter == 6:
                        to_print += ' '
                        counter = -1
                elif '.' in s:
                    a, b = s.split('.')
                    to_print += s
                    counter = len(b)
            print(to_print)
    # print(len(l.split(' ')))
