-- <@> JewelSuite(TM) ECLIPSE Deck Builder

-- <+> Start of deck ECL

NOECHO
-- <+> RUNSPEC Section

RUNSPEC
TITLE
VEM

-- ascii formatted output
--FMTOUT
--UNIFOUT

DIMENS
    60 60 7  /

METRIC
OIL
WATER
NUMRES
    1 /

TABDIMS
    2*    24 2*    20    20 1*     1 7* /
EQLDIMS
    2* 100 2* /
REGDIMS
    6* /
WELLDIMS
       12   100     4    12     0     0     0     0     0     0     0     0 /
VFPPDIMS
    6* /
VFPIDIMS
    3* /
AQUDIMS
    2*     1 3* /
NSTACK
 -75 /
START
15 JUN 2011 /

-- <-> RUNSPEC Section

-- <+> GRID Section

GRID 


SPECGRID
    60 60 7 1 F /


INCLUDE
   '../ACTIVE.INC' /


DX
    25200*8 /


DY
    25200*8 /


DZ
    25200*4 /

TOPS
    3600*4000 3600*4004 3600*4008 3600*4012 3600*4016 3600*4020 3600*4024/

INCLUDE
'../mDARCY.INC'
/

NTG
  25200*1 /

PORO
    25200*0.2 /

ECHO

INIT
/

-- <-> GRID Section


-- <+> PROPS Section

PROPS

DENSITY
     900 1000          1 /
PVCDO
    400          1 1.000E-05          5          0
    /

PVTW
    400          1 1.000E-05        1          0 /

ROCK
    400          0 /

SWOF
      0.1000,  0.0000e+00,  8.0000e-01  0
      0.2000,  0,  8.0000e-01  0
      0.2500,  2.7310e-04,  5.8082e-01  0
	0.3000,  2.1848e-03,  4.1010e-01  0
	0.3500,  7.3737e-03,  2.8010e-01  0
	0.4000,  1.7478e-02,  1.8378e-01  0
	0.4500,  3.4138e-02,  1.1473e-01  0
	0.5000,  5.8990e-02,  6.7253e-02  0
	0.5500,  9.3673e-02,  3.6301e-02  0
	0.6000,  1.3983e-01,  1.7506e-02  0
	0.6500,  1.9909e-01,  7.1706e-03  0
	0.7000,  2.7310e-01,  2.2688e-03  0
	0.7500,  3.6350e-01,  4.4820e-04  0
	0.8000,  4.7192e-01,  2.8000e-05  0
	0.8500,  6.0000e-01,  0.0000e+00  0
	0.9000,  7.4939e-01,  0.0000e+00  0
	       
/



-- <-> PROPS Section

-- <+> REGIONS Section

REGIONS

-- <-> REGIONS Section

-- <+> SOLUTION Section

SOLUTION
EQUIL
       4000  400    5000          0 /

--RPTSOL
--    RESTART=2 FIP=3/


RPTSOL
 RESTART=2 /
/

-- <-> SOLUTION Section

-- <+> SUMMARY Section

SUMMARY
FOPR
FWPR
FWIR
WOPR
'PROD1'
'PROD2'
'PROD3'
'PROD4'
/
WWPR
'PROD1'
'PROD2'
'PROD3'
'PROD4'
/
WWIR
'INJECT1' 
'INJECT2'
'INJECT3' 
'INJECT4' 
'INJECT5'
'INJECT6'
'INJECT7'
'INJECT8'
/   
FOPT
FWPT
FWIT
WLPR
'PROD1'
'PROD2'
'PROD3'
'PROD4'
/
--BRPV 
--1 1 1 /
EXCEL

--ALL

RPTONLY

-- <-> SUMMARY Section

-- <+> SCHEDULE Section

SCHEDULE


-- <+> SCHEDULE 7/7/2011 (0 days)

RPTSCHED
    FIP WELSPECS WELLS /

RPTRST
   BASIC=2/

WELSPECS
    	'INJECT1' '1'   5    57  1* 'WATER' /
	'INJECT2' '1'   30   53  1* 'WATER' /
	'INJECT3' '1'   2    35  1* 'WATER' /
	'INJECT4' '1'   27   29  1* 'WATER' /
	'INJECT5' '1'   50   35  1* 'WATER' /
	'INJECT6' '1'   8    9   1* 'WATER' /
	'INJECT7' '1'   32   2   1* 'WATER' /
	'INJECT8' '1'   57   6   1* 'WATER' /
    	'PROD1'   '1'   16   43  1* 'OIL' /
    	'PROD2'   '1'   35   40  1* 'OIL' /
    	'PROD3'   '1'   23   16  1* 'OIL' /
    	'PROD4'   '1'   43   18  1* 'OIL' /
/

COMPDAT
    	'INJECT1'    2*    1     7 'OPEN' 2*     0.2 	1*          0 / 
	'INJECT2'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
	'INJECT3'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
	'INJECT4'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
	'INJECT5'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
	'INJECT6'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
	'INJECT7'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
	'INJECT8'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    	'PROD1'      2*    1     7 'OPEN' 2*     0.2 	1*          0 / 
    	'PROD2'      2*    1     7 'OPEN' 2*     0.2 	1*          0 / 
    	'PROD3'      2*    1     7 'OPEN' 2*     0.2 	1*          0 / 
    	'PROD4'      2*    1     7 'OPEN' 2*     0.2	1*          0 / 
/

TUNING
0.1 30 /
/
12 1 250 1* 25 /

WCONPROD
    'PROD1' 'OPEN' 'BHP' 5* ${300}/
    'PROD3' 'OPEN' 'BHP' 5* ${300}/
    'PROD2' 'OPEN' 'BHP' 5* ${300}/
    'PROD4' 'OPEN' 'BHP' 5* ${300}/
/

WCONINJE
'INJECT1' 'WATER' 'OPEN' 'RATE' ${injrate[0]} 1* 420/
'INJECT2' 'WATER' 'OPEN' 'RATE' ${injrate[1]} 1* 420/
'INJECT3' 'WATER' 'OPEN' 'RATE' ${injrate[2]} 1* 420/
'INJECT4' 'WATER' 'OPEN' 'RATE' ${injrate[3]} 1* 420/
'INJECT5' 'WATER' 'OPEN' 'RATE' ${injrate[4]} 1* 420/
'INJECT6' 'WATER' 'OPEN' 'RATE' ${injrate[5]} 1* 420/
'INJECT7' 'WATER' 'OPEN' 'RATE' ${injrate[6]} 1* 420/
'INJECT8' 'WATER' 'OPEN' 'RATE' ${injrate[7]} 1* 420/
/

TSTEP
   12*30 /
   
WCONPROD
    'PROD1' 'OPEN' 'BHP' 5* ${300}/
    'PROD3' 'OPEN' 'BHP' 5* ${300}/
    'PROD2' 'OPEN' 'BHP' 5* ${300}/
    'PROD4' 'OPEN' 'BHP' 5* ${300}/
/

WCONINJE
'INJECT1'	'WATER'	'OPEN'	'RATE'	${injrate[8]} 1* 420/
'INJECT2'	'WATER'	'OPEN'	'RATE'	${injrate[9]} 1* 420/
'INJECT3'	'WATER'	'OPEN'	'RATE'	${injrate[10]} 1* 420/
'INJECT4'	'WATER'	'OPEN'	'RATE'	${injrate[11]} 1* 420/
'INJECT5'	'WATER'	'OPEN'	'RATE'	${injrate[12]} 1* 420/
'INJECT6'	'WATER'	'OPEN'	'RATE'	${injrate[13]} 1* 420/
'INJECT7'	'WATER'	'OPEN'	'RATE'	${injrate[14]} 1* 420/
'INJECT8'	'WATER'	'OPEN'	'RATE'	${injrate[15]} 1* 420/
/
   
TSTEP
   12*30 /

WCONPROD
    'PROD1' 'OPEN' 'BHP' 5* ${300}/
    'PROD3' 'OPEN' 'BHP' 5* ${300}/
    'PROD2' 'OPEN' 'BHP' 5* ${300}/
    'PROD4' 'OPEN' 'BHP' 5* ${300}/
/

WCONINJE
'INJECT1'	'WATER'	'OPEN'	'RATE'	${injrate[16]} 1* 420/
'INJECT2'	'WATER'	'OPEN'	'RATE'	${injrate[17]} 1* 420/
'INJECT3'	'WATER'	'OPEN'	'RATE'	${injrate[18]} 1* 420/
'INJECT4'	'WATER'	'OPEN'	'RATE'	${injrate[19]} 1* 420/
'INJECT5'	'WATER'	'OPEN'	'RATE'	${injrate[20]} 1* 420/
'INJECT6'	'WATER'	'OPEN'	'RATE'	${injrate[21]} 1* 420/
'INJECT7'	'WATER'	'OPEN'	'RATE'	${injrate[22]} 1* 420/
'INJECT8'	'WATER'	'OPEN'	'RATE'	${injrate[23]} 1* 420/
/
   
TSTEP
   12*30 /
   
WCONPROD
    'PROD1' 'OPEN' 'BHP' 5* ${300}/
    'PROD3' 'OPEN' 'BHP' 5* ${300}/
    'PROD2' 'OPEN' 'BHP' 5* ${300}/
    'PROD4' 'OPEN' 'BHP' 5* ${300}/
/

WCONINJE
'INJECT1'	'WATER'	'OPEN'	'RATE'	${injrate[24]} 1* 420/
'INJECT2'	'WATER'	'OPEN'	'RATE'	${injrate[25]} 1* 420/
'INJECT3'	'WATER'	'OPEN'	'RATE'	${injrate[26]} 1* 420/
'INJECT4'	'WATER'	'OPEN'	'RATE'	${injrate[27]} 1* 420/
'INJECT5'	'WATER'	'OPEN'	'RATE'	${injrate[28]} 1* 420/
'INJECT6'	'WATER'	'OPEN'	'RATE'	${injrate[29]} 1* 420/
'INJECT7'	'WATER'	'OPEN'	'RATE'	${injrate[30]} 1* 420/
'INJECT8'	'WATER'	'OPEN'	'RATE'	${injrate[31]} 1* 420/
/
   
TSTEP
   12*30 /
   
WCONPROD
    'PROD1' 'OPEN' 'BHP' 5* ${300}/
    'PROD3' 'OPEN' 'BHP' 5* ${300}/
    'PROD2' 'OPEN' 'BHP' 5* ${300}/
    'PROD4' 'OPEN' 'BHP' 5* ${300}/
/

WCONINJE
'INJECT1'	'WATER'	'OPEN'	'RATE'	${injrate[32]} 1* 420/
'INJECT2'	'WATER'	'OPEN'	'RATE'	${injrate[33]} 1* 420/
'INJECT3'	'WATER'	'OPEN'	'RATE'	${injrate[34]} 1* 420/
'INJECT4'	'WATER'	'OPEN'	'RATE'	${injrate[35]} 1* 420/
'INJECT5'	'WATER'	'OPEN'	'RATE'	${injrate[36]} 1* 420/
'INJECT6'	'WATER'	'OPEN'	'RATE'	${injrate[37]} 1* 420/
'INJECT7'	'WATER'	'OPEN'	'RATE'	${injrate[38]} 1* 420/
'INJECT8'	'WATER'	'OPEN'	'RATE'	${injrate[39]} 1* 420/
/
   
TSTEP
   12*30 /
   
WCONPROD
    'PROD1' 'OPEN' 'BHP' 5* ${300}/
    'PROD3' 'OPEN' 'BHP' 5* ${300}/
    'PROD2' 'OPEN' 'BHP' 5* ${300}/
    'PROD4' 'OPEN' 'BHP' 5* ${300}/
/

WCONINJE
'INJECT1'	'WATER'	'OPEN'	'RATE'	${injrate[40]} 1* 420/
'INJECT2'	'WATER'	'OPEN'	'RATE'	${injrate[41]} 1* 420/
'INJECT3'	'WATER'	'OPEN'	'RATE'	${injrate[42]} 1* 420/
'INJECT4'	'WATER'	'OPEN'	'RATE'	${injrate[43]} 1* 420/
'INJECT5'	'WATER'	'OPEN'	'RATE'	${injrate[44]} 1* 420/
'INJECT6'	'WATER'	'OPEN'	'RATE'	${injrate[45]} 1* 420/
'INJECT7'	'WATER'	'OPEN'	'RATE'	${injrate[46]} 1* 420/
'INJECT8'	'WATER'	'OPEN'	'RATE'	${injrate[47]} 1* 420/
/
   
TSTEP
   12*30 /
   
WCONPROD
    'PROD1' 'OPEN' 'BHP' 5* ${300}/
    'PROD3' 'OPEN' 'BHP' 5* ${300}/
    'PROD2' 'OPEN' 'BHP' 5* ${300}/
    'PROD4' 'OPEN' 'BHP' 5* ${300}/
/

WCONINJE
'INJECT1'	'WATER'	'OPEN'	'RATE'	${injrate[48]} 1* 420/
'INJECT2'	'WATER'	'OPEN'	'RATE'	${injrate[49]} 1* 420/
'INJECT3'	'WATER'	'OPEN'	'RATE'	${injrate[50]} 1* 420/
'INJECT4'	'WATER'	'OPEN'	'RATE'	${injrate[51]} 1* 420/
'INJECT5'	'WATER'	'OPEN'	'RATE'	${injrate[52]} 1* 420/
'INJECT6'	'WATER'	'OPEN'	'RATE'	${injrate[53]} 1* 420/
'INJECT7'	'WATER'	'OPEN'	'RATE'	${injrate[54]} 1* 420/
'INJECT8'	'WATER'	'OPEN'	'RATE'	${injrate[55]} 1* 420/
/
   
TSTEP
   12*30 /
   
WCONPROD
    'PROD1' 'OPEN' 'BHP' 5* ${300}/
    'PROD3' 'OPEN' 'BHP' 5* ${300}/
    'PROD2' 'OPEN' 'BHP' 5* ${300}/
    'PROD4' 'OPEN' 'BHP' 5* ${300}/
/

WCONINJE
'INJECT1'	'WATER'	'OPEN'	'RATE'	${injrate[56]} 1* 420/
'INJECT2'	'WATER'	'OPEN'	'RATE'	${injrate[57]} 1* 420/
'INJECT3'	'WATER'	'OPEN'	'RATE'	${injrate[58]} 1* 420/
'INJECT4'	'WATER'	'OPEN'	'RATE'	${injrate[59]} 1* 420/
'INJECT5'	'WATER'	'OPEN'	'RATE'	${injrate[60]} 1* 420/
'INJECT6'	'WATER'	'OPEN'	'RATE'	${injrate[61]} 1* 420/
'INJECT7'	'WATER'	'OPEN'	'RATE'	${injrate[62]} 1* 420/
'INJECT8'	'WATER'	'OPEN'	'RATE'	${injrate[63]} 1* 420/
/
   
TSTEP
   12*30 /
   
WCONPROD
    'PROD1' 'OPEN' 'BHP' 5* ${300}/
    'PROD3' 'OPEN' 'BHP' 5* ${300}/
    'PROD2' 'OPEN' 'BHP' 5* ${300}/
    'PROD4' 'OPEN' 'BHP' 5* ${300}/
/

WCONINJE
'INJECT1'	'WATER'	'OPEN'	'RATE'	${injrate[64]} 1* 420/
'INJECT2'	'WATER'	'OPEN'	'RATE'	${injrate[65]} 1* 420/
'INJECT3'	'WATER'	'OPEN'	'RATE'	${injrate[66]} 1* 420/
'INJECT4'	'WATER'	'OPEN'	'RATE'	${injrate[67]} 1* 420/
'INJECT5'	'WATER'	'OPEN'	'RATE'	${injrate[68]} 1* 420/
'INJECT6'	'WATER'	'OPEN'	'RATE'	${injrate[69]} 1* 420/
'INJECT7'	'WATER'	'OPEN'	'RATE'	${injrate[70]} 1* 420/
'INJECT8'	'WATER'	'OPEN'	'RATE'	${injrate[71]} 1* 420/
/
   
TSTEP
   12*30 /
   
WCONPROD
    'PROD1' 'OPEN' 'BHP' 5* ${300}/
    'PROD3' 'OPEN' 'BHP' 5* ${300}/
    'PROD2' 'OPEN' 'BHP' 5* ${300}/
    'PROD4' 'OPEN' 'BHP' 5* ${300}/
/

WCONINJE
'INJECT1'	'WATER'	'OPEN'	'RATE'	${injrate[72]} 1* 420/
'INJECT2'	'WATER'	'OPEN'	'RATE'	${injrate[73]} 1* 420/
'INJECT3'	'WATER'	'OPEN'	'RATE'	${injrate[74]} 1* 420/
'INJECT4'	'WATER'	'OPEN'	'RATE'	${injrate[75]} 1* 420/
'INJECT5'	'WATER'	'OPEN'	'RATE'	${injrate[76]} 1* 420/
'INJECT6'	'WATER'	'OPEN'	'RATE'	${injrate[77]} 1* 420/
'INJECT7'	'WATER'	'OPEN'	'RATE'	${injrate[78]} 1* 420/
'INJECT8'	'WATER'	'OPEN'	'RATE'	${injrate[79]} 1* 420/
/
   
TSTEP
   12*30 /

-- END OF DECK 

-- <@> JewelSuite(TM) ECLIPSE Deck Builder
