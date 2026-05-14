
with open(events_path, "rb") as f:
            events = pickle.load(f)


print(events[:10])

[[39869     3     0     1]
 [46491     0     0     1]
 [53010     3     0     1]
 [59649     3     0     1]
 [66270     2     0     1]
 [73063     1     0     1]
 [79838     3     0     1]
 [86528     1     0     1]
 [93064     3     0     1]
 [99823     1     0     1]]



 [[39869     3     0     1]
#  col0   col1  col2  col3

events[:, 0]  # amostra/tempo do evento
events[:, 1]  # rótulo da classe: 0, 1, 2, 3
events[:, 2]  # talvez condição extra
events[:, 3]  # condição da tarefa: inner = 1