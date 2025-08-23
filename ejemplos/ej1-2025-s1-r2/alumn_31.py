def validar_texto(texto):
  e=0
  while e<len(texto):
    if 'a'<=texto[e]<='z'or 'A'<=texto[e]<='Z' or '0'<=texto[e]<='9' or texto[e]==' ' or texto[e]=='\n':
      
      if e+1==(len(texto)):
        return True
      else:
        e+=1
  
    else:
      return False
 
 
def mostrar_lineas(texto):
  if validar_texto(texto)==True:
    respuesta=''
    lista=[]
    i=0
    e=0
    n=1
    while i<len(texto)
      if letra[i]!=\n:
        respuesta+=letra[i]
      elif i==len(texto)-1:
        lista.append(respuesta)
      else:
        lista.append(respuesta)
        respuesta=''
      
    while n<=len(lista)
      print(n,':',lista[e])
      n+=1
      e+=1
    
  else:
    print('Texto inválido')
    
def promedio_numeros(texto): 
  if validar_texto(texto)==True:
    cantidaddenums=0
    sumadenums=0
    i=0
    e=0
    while i<len(texto):
      if texto[i]==' ' or texto[i]==\n:
        if '0'<=texto[i+1]<='9':
          e=i+1
          while '0'<=texto[e]<='9':
            
        
        
    
  else:
    print('Texto inválido')
    
 
 
print(validar_texto(texto))
mostrar_lineas(texto)
promedio_numeros(texto)