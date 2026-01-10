#aclaracion: por comodidad utilice notasp1 y notasp2 como parametros en vez de notas2p y notas1p
def recursan(nombres, apellidos, notasp1, notasp2):
  print('alumnos que recursan la materia: ')
  contador =0
  for i in range(len(nombres)):
    promedio = (notasp1[i]+notasp2[i])/2:#me tira error de sintaxis, pero lo que estoy planteando tiene sentido
      if promedio <=3:
        print(nombres[i],'-Promedio: ', round(promedio,2))
        contador +=1
        if contador%3 == 0:
          print('----------------------------')
def notas_ap_largo(apellidos, notasp2):
  print('Notas del 2do parcial de apellidos más largos: ')
  max_len = len(apellidos[0])
  for j in range(len(apellidos)):
    if len(apellidos[j])>max_len:
      max_len = len(apellidos[j])
      print(apellidos[j], '-Nota: ', notasp2[j])
def revision(nombres, apellidos, notasp1, notasp2):
  print('Alumnos que se debe revisar posible promoción: ')
  for i in range(len(nombres)):
    promedio_rev = (notasp1[i]+notasp2[i])/2
    if (notasp1[i]<7 and notasp2[2]>=6) or (notasp1[i]>=6 and notasp2[i]<7) and promedio_rev>=7:#supongo que la nota de un parcial es mayot a 7 y la nota del otro parcial es menos igual a 6
      for m in range(len(nombres)):
        for l in range(m+1,len(nombres)):
          if nombres[m]<nombres[l]:
            nombres[m], nombres[l] = nombres[m], nombres[l]
            print(nombres[m],nombres[l])
def mostrar_cursada(nombres, apellidos, notasp1, notasp2):
  if not nombres:
    print('no hay notas cargadas')
    return
  if not apellidos:
    print('no hay notas cargadas')
    return
  recursan(nombres, apellidos, notasp1, notasp2)
  notas_ap_largo(apellidos, notasp2)
  revision(nombres, apellidos, notasp1, notasp2)
 
 
 
 
nombres = ["Ana", "Juan", "Luis", "María", "Lucía", "Ruben", "Adrian", "Jorge"]
apellidos = ["Li", "Gómez", "Paz", "Sosa", "Ro", "Paz", "Martinez", "Carranza"]
notasp1 = [1, 2, 0, 8, 6.50, 9, 9, 1]
notasp2 = [1, 1.50, 3, 7, 9, 7, 10, 5]  
mostrar_cursada(nombres, apellidos, notasp1, notasp2)
 
 