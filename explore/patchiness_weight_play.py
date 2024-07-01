#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:12:35 2024

@author: anna
"""
from scipy.special import gamma

vol_mult = (2*(np.pi**(768/2)))/(768*gamma(768/2))

m = sum((np.array(dist4_norm)**768)*hist4)/quant4.k
V = np.var((np.array(dist4_norm)**768)*hist4)
m_star = m +((V/m)-1) #variance adjusted density
m_star/m

g = dist4_norm**768
max(g)
np.median(g)

