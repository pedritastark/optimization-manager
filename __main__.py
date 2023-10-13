#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:11:45 2023

@author: sebastianpedraza

Optimizacion en una variable
"""


import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

class OptimizerManager:
    def __init__(self):
        self.optimizers = {}
        self.algorithm = None

    def register_optimizer(self, name, optimizer_function):
        self.optimizers[name] = optimizer_function

    def optimize(self, algorithm_name, *args, **kwargs):
        if algorithm_name in self.optimizers:
            optimizer_function = self.optimizers[algorithm_name]
            return optimizer_function(*args, **kwargs)
        else:
            raise ValueError(f"El metodo '{algorithm_name}' no esta registrado")

    

    ## Metodo para graficar la funcion
    def GraphObjectiveFunction(self):
        
        'Usamos esta funcion para graficar la funcion que estamos evaluando'
        'a y b son los limites a los cuales esta regida la funcion [a,b]'
        
        a, b = -10, 10    
        T = np.linspace(a, b, 100)
        U = 3*(T**4) + 4*(T**3)

        plt.figure(figsize=(6, 3))
        plt.plot(T, U, 'b')
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid()
        plt.show()
        
        
    '''    
    U(T) es una funcion objetivo  que usamos para encontrar el minimo
    usando los algoritmos algoritmos por  ejemplo en el algoritmo de busqueda
    de la seccion dorada para ir reduciendo el intervalo donde se encuentra la solucion optima
    el argumento T es sobre el cual evaluamos la funcion objetivo en ese punto (T, f(T))
    '''

    def U(self, T):
        return sp.exp(-T) + sp.log(T**2 + 1)

    def derU(self, T):
        return 2*T/(T**2 + 1) - np.exp(-T)

    def der2U(self, T):
        return -4*T**2/(T**2 + 1)**2 + np.exp(-T) + 2/(T**2 + 1)
    
    ## Metodo de newton
    def newtonSearch(self):
        epsilon   = 0.0001
        lambda_k  = -2   
        k         = 1
        
        while True:
            if abs(self.derU(lambda_k)) < epsilon:
                print("-------------------------------------------------------")
                print("Punto óptimo {}".format(lambda_k))
                break
            else:
                lambda_k_1 = lambda_k - (self.derU(lambda_k) / self.der2U(lambda_k))
                if abs(lambda_k_1 - lambda_k) < epsilon:
                    print("-------------------------------------------------------")
                    print("Punto óptimo {}".format(lambda_k_1))
                    break
                else: 
                    lambda_k = lambda_k_1
            k += 1
            

    ## Metodo de la seccion dorada    
    def GoldenSectionSearch(self):
        
    

    
        '''
        Definimos  [a,b] el intervalo incial de busqueda, alpha la constante 
        sobre la cual calculamos lambda_k y miu_k, h es la constante 
        sobre la cual definimos el espacio entre el intervalo encontrado 
        '''
        a       =  -2
        b       =  2
        alpha   =  (5**(1/2)-1)/2
        h       =  0.001
        
        '''
        el contador solo lo usamos para mostrar el numero de iteraciones que hemos realizado
        data guarda los valores de el numero de iteracion, lambda y f(lambda)
        '''
        
        cont = 0

        while(True):
            
            'calculamos alpha 1 y miu'
            cont    =  cont + 1
            lambda1 =  a + (1-alpha) * (b-a)
            miu1    =  a + alpha*(b-a)

            'calulmamos f(alpha1) y f(alpha2)'
            U_lambda1 = self.U(lambda1)
            U_miu1 = self.U(miu1)
            
            'como f(lambda) > f(miu), entonces el limite sobre el intervalo [a,b] pasa a ser'
            '[lambda, b]'
            if(U_lambda1 > U_miu1):
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda1, miu1, U(lambda1), U(miu1), b-a))
                a = lambda1
                
                'de otro modo el limite se reduce el intervalo [a, miu]'
            else:
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda1, miu1, U(lambda1), U(miu1), b-a))

                b = miu1       

            'Mostramos numero de iteracion y el intervalo al cual vamos reduciendo'
            print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}".format(cont, a, b))

            if(np.abs(a - b) < h):
                
                'Mostramos el intervalo final luego de que este cumpla con el maximo'
                'de distancia entre los puntos a y b sea menos a h'
                
                print("-------------------------------------------------------")
                print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}".format(cont, a, b))
                
                'Como la solucion optima pertenece al intervalo [a,b]'
                'retornamos (a+b) / 2'

                print("(a_k + b_k)/2 == {}".format((a + b)/2))
                break
            

    ## Metodo de biseccion
    def bisectionSearch(self):
        '''
        Definimos  [a,b] el intervalo incial de busqueda,h es la constante 
        sobre la cual definimos el espacio entre el intervalo (intervalo de incertidumbre),
        cont es el contador para mostar el numero de iteracion
        '''
        a    = -4
        b    = 4
        h    = 0.0001
        cont = 0
        while True:
            cont += 1
            
            
            if b-a < h:
                
                'Mostramos el intervalo final luego de que este cumpla con el maximo'
                'de distancia entre los puntos a y b sea menos a h'
                
                print("-------------------------------------------------------")
                print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}".format(cont, a, b))
                
                'Como la solucion optima pertenece al intervalo [a,b]'
                'retornamos (a+b) / 2'
                print("(a_k + b_k)/2 == {}".format((a + b)/2))
                break
            
            'Calculamos lambda_k' 
            lambda_k = (a+b)/2 
            
            'Como el algoritmo se aplica sobre funciones pseudoconvexa'
            'si la derivada en le punto lambda_k es 0, entonces estamos en le minimo'
            
            if self.derU(lambda_k) == 0:
                print("-------------------------------------------------------")
                print("Solucion optima == {}".format(lambda_k))
                
                
                'Si la derivada es >0 evaluada en lambda_k como es una funcion pseudoconvexa'
                'el nuevo intervalo es [a, lambda_k]'
            elif self.derU(lambda_k) > 0: 
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f} - l_k: {:.10f} - f(l_k): {:.10f} - a_k-b_k: {:.10f}".format(cont, a, b, lambda_k, derU(lambda_k), b-a))
                b = lambda_k
                
                
                'Si la derivada es <0 evaluada en lambda_k como es una funcion pseudoconvexa'
                'el nuevo intervalo es [lambda_k, b]'
            else: 
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f} - l_k: {:.10f} - f(l_k): {:.10f} - a_k-b_k: {:.10f}".format(cont, a, b, lambda_k, derU(lambda_k), b-a))
                a = lambda_k
            print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}".format(cont, a, b))
            
    
    # Metodo de Fibonacci
    
    def fibonacci(self,n):
        '''
        Fibonnaci retorna el resultado de la sucesion de fibonacci usado en el metodo de busqueda 
        para encontrar  el numero de iteraciones que vamos a hacer para encontrar el 
        intervalo donde esta la  solucion optima
        '''
    
        if n<= 1: return 1 
        else: return self.fibonacci(n-1) + self.fibonacci(n-2)
    def iterationFinder(self,a,b,h):
        cv = (b-a)/h
        i = 0
        while True:
            if self.fibonacci(i) > cv: return i
            i+=1    
        
    
    def fibonacciSearch(self):
        '''
        Definimos  [a,b] el intervalo incial de busqueda,h la longitud final
        de incertidumbre y cont para llevar contador para mostar el numero de iteracion,
        e que es un epsilon > 0 que es un constante de distinguibilidad que cumple con
        e < h y n la cantidad de iteraciones
        '''

        a    =  -4
        b    =  2
        e    =  0.01
        h    =  0.1
        cont =  1        
        n    =  self.iterationFinder(a, b, h)  
        
        while True:
            
            'Calculamos lambda_k y miu_k, y llevamos el contador de iteraciones'
            lambda_k =  a + (1-(self.fibonacci(n-1)/self.fibonacci(n)))*(b-a) 
            miu_k    =  a + (self.fibonacci(n-1)/self.fibonacci(n))*(b-a)
            cont    +=  1
            
            'si f(lambda_k) > f(miu_k) ajustamos el valor del intervalo a [lambda_k, b_k]'
            if self.U(lambda_k) > self.U(miu_k): 
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda_k, miu_k, U(lambda_k), U(miu_k), b-a))
                a = lambda_k
                
                'si f(lambda_k) <= f(miu_k) ajustamos el valor del intervalo a [a_k, miu_k]'
            else:
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda_k, miu_k, U(lambda_k), U(miu_k), b-a))
                b = miu_k
            
            print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}".format(cont, a, b))


            ## condicional para cuando llegamos a la iteracion n-2 
            if cont == n-2: break
                
                
        miu_k = lambda_k + e
        
        '''
        Mostramos el intervalo final luego de que este cumpla con el maximo
        de distancia entre los puntos a y b sea menos a h
        '''
        
        if self.U(lambda_k) > self.U(miu_k): 
            # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda_k, miu_k, U(lambda_k), U(miu_k), b-a))
            a = lambda_k
            
            "'si f'(lambda_k) <= f'(miu_k) ajustamos el valor del intervalo a [a_k, miu_k]"
        else:
            # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda_k, miu_k, U(lambda_k), U(miu_k), b-a))
            b = miu_k
        
        
        print("-------------------------------------------------------")
        '''
        Como la solucion optima pertenece al intervalo [a,b]
        retornamos (a+b) / 25
        '''
        print("(a_k + b_k)/2 == {}".format((a + b)/2))

            





# Crear una instancia de la clase OptimizerManager y registra los metodos de optimizacion
optimizer_manager = OptimizerManager()
optimizer_manager.register_optimizer("GoldenSectionSearch", optimizer_manager.GoldenSectionSearch)
optimizer_manager.register_optimizer("Newton", optimizer_manager.newtonSearch)
optimizer_manager.register_optimizer("Bisection", optimizer_manager.bisectionSearch)
optimizer_manager.register_optimizer("Fibonacci", optimizer_manager.fibonacciSearch)





optimizer_manager.algorithm = "Fibonacci"

# Utilizar el algoritmo de optimización y el gráfico
optimizer_manager.optimize(optimizer_manager.algorithm)
optimizer_manager.GraphObjectiveFunction()