# Opmerkingen

.state_dict[]
model.modules()   !!
toch.nn.conv2d
alles incapsuleren in: with torch.nograd
for m in model.modules()
	mm = m.weight()

m.weight.data = new_w
biassen aanpassen 1 uitsmijten

pytorch computation graph
jit trace

als alternatief als graph niet gemaakt kan worden, laatste conv niet prunen.

device = net.device