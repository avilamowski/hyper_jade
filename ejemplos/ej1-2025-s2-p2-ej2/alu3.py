import csv
def costo(inventario_csv, venta_txt):
  try:
    with open(venta_txt, "r") as a:
      ctxt=a.read()
  except FileNotFoundError:
    print("El archivo no existe")
  try:
    with open(inventario_csv) as s:
      ccsv=s.read()#Asumo que estoy leyendo lo que esta en el archivo como si fuera un texto, aunque sean listas yo lo tomo como un texto cada linea
  except FileNotFoundError:
    print("El archivo no existe")
  cosas=""
  listatxt=[]
  i=0
  while i<len(ctxt):
    if ctxt[i]!= "\n" or ctxt!=":":
      cosas+=ctxt[i]
    elif ctxt[i]== "\n" or ctxt==":":
      listatxt.append(cosas)
    elif i==len(ctxt)-1:
      listatxt.append(cosas)
    i+=1
  j=0
  cosascsv=""
  listacsv=[]
  while j<len(ccsv):
    if ccsv[j]!= "\n" or ccsv!=",":
      cosascsv+=ccsv[j]
    elif ccsv[j]== "\n" or ccsv==",":
      listatxt.append(cosascsv)
    elif j==len(ccsv)-1:
      listacsv.append(cosascsv)
  x=0
  total=0.0
  t=#posicion en la que esta el articulo en listacsv
  while x<len(listatxt):
    if listatxt[x] in listacsv:#Asumo que de alguna manera tengo la posicion en la listacsv en la que se encuentra ese articulo que esta en listatxt(no llegue con el tiempo a)
      if (listacsv[t+2]-listatxt[x+1])>=0:
        total+=listatxt[x+1]*listacsv[t+1]
        listacsv[t+2]=listacsv[t+2]-listatxt[x+1]
      elif (listacsv[t+2]-listatxt[x+1])<0:
        t=listacsv[t+2]
        total+=t*listacsv[t+1]
      elif listacsv[t+2]==0:
        total+=0
    x+=2
    
   return total
 
