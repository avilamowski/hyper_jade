def mostrar_cursada(nombres, apellidos, notas1p, notas2p):
    if nombres == [] or apellidos == [] or notas1p == [] or notas2p == []:
        print("No hay notas cargadas")
    else:
        contador = 3
        alguien = 0
        for i in range(len(nombres)):
            persona = ""
            promedio = (notas1p[i] + notas2p[i]) / 2
            if contador == 0:
                contador = 3
                print("----------------------------")
            if ((notas1p[i] + notas2p[i]) / 2) < 4:
                persona += nombres[i] + " " + apellidos[i]
                contador -= 1
                alguien += 1
                if alguien != 1:
                    print(f"{persona} - Promedio: {promedio}")
                if alguien == 1:
                    print("Alumnos que recursan la materia: ")
                    print(f"{persona} - Promedio: {promedio}")
        notaapellidomaslargo = [0]
        z = 0
        apellidomaslargo = [""]
        apellidomaslargos = ""
        for p in apellidos:
            chequeo = ""
            chequeo += p
            if len(chequeo) > len(apellidomaslargos):
                apellidomaslargos = chequeo
                apellidomaslargo[0] = apellidomaslargos
                notaapellidomaslargo[0] = notas2p[z]
            elif len(chequeo) == len(apellidomaslargos):
                apellidomaslargo.append(chequeo)
                notaapellidomaslargo.append(notas2p[z])
            z += 1
        if len(apellidomaslargo) > 1:
            print("\nNotas del 2do parcial de apellidos mas largos: ")
            a = 0
            ultimo = [""]
            ultimanota = [0]
            ultimos = ""
            for y in apellidomaslargo:
                porfa = ""
                porfa += y
                if len(porfa) > len(ultimos):
                    ultimos = porfa
                    ultimo[0] = ultimos
                    ultimanota[0] = notaapellidomaslargo[a]
                elif len(porfa) == len(ultimos):
                    ultimo.append(porfa)
                    ultimanota.append(notaapellidomaslargo[a])
                a += 1
            quien = ""
            for q in range(len(ultimo)):
                quien = ultimo[q]
                print(f"{quien} - Nota2p: {ultimanota[q]}")

        else:
            print("\nNotas del 2do parcial de apellidos mas largos: ")
            print(f"{apellidomaslargo} - Nota 2p: {notaapellidomaslargo}")

        enrevision = []
        apellidoenrevision = []
        for x in range(len(nombres)):
            revision = (notas1p[x] + notas2p[x]) / 2
            if revision > 7 and (7 > notas1p[x] >= 6 or 7 > notas2p[x] >= 6):
                enrevision.append(nombres[x])
                apellidoenrevision.append(apellidos[x])
        for l in range(len(enrevision)):
            for b in range(len(enrevision) - 1 - l):
                if enrevision[b] < enrevision[b + 1]:
                    enrevision[b], enrevision[b + 1] = enrevision[b + 1], enrevision[b]
                    apellidoenrevision[b], apellidoenrevision[b + 1] = (
                        apellidoenrevision[b + 1],
                        apellidoenrevision[b],
                    )
        print("\nAlumnos que se debe revisar posible promocion: \n")
        arevision = ""
        for u in range(len(enrevision)):
            arevision += enrevision[u] + " " + apellidoenrevision[u]
            print(arevision)
            arevision = ""
