      SUBROUTINE USDFLD(FIELD,STATEV,PNEWDT,DIRECT,T,CELENT,
     1 TIME,DTIME,CMNAME,ORNAME,NFIELD,NSTATV,NOEL,NPT,LAYER,
     2 KSPT,KSTEP,KINC,NDI,NSHR,COORD,JMAC,JMATYP,MATLAYO,
     3 LACCFLA)
C
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME,ORNAME
      CHARACTER*3  FLGRAY(15)
      DIMENSION FIELD(NFIELD),STATEV(NSTATV),DIRECT(3,3),
     1 T(3,3),TIME(2)
      DIMENSION ARRAY(15),JARRAY(15),JMAC(*),JMATYP(*),
     1 COORD(*)
C
      REAL STRAIN_RATE, SR
      REAL EPS, EPS0, EPS_OLD, SR_OLD, DTIME_OLD
      REAL K_VAL, M_VAL, PLM, ESO
      REAL K_MIN, M_MIN
C
C ----------------------------------------------------------------------
C Material model:
C     sigma = K(eps) * epsdot ** m(eps)
C
C Abaqus power law form used through FIELD variables:
C     epsdot = PLM * sigma ** ESO
C
C Therefore:
C     ESO = 1 / m
C     PLM = K ** (-1 / m)
C ----------------------------------------------------------------------
C
C Read current equivalent creep/plastic strain.
C In the attached analogue ARRAY(7) was used for CE.
      CALL GETVRM('CE',ARRAY,JARRAY,FLGRAY,JRCD,JMAC,JMATYP,
     1 MATLAYO,LACCFLA)
      EPS = ARRAY(7)
C
C Read previous state variables:
C STATEV(3) = previous strain
C STATEV(4) = previous strain rate
C STATEV(5) = previous time increment
      CALL GETVRM('SDV',ARRAY,JARRAY,FLGRAY,JRCD,JMAC,JMATYP,
     1 MATLAYO,LACCFLA)
      EPS_OLD = ARRAY(3)
      SR_OLD = ARRAY(4)
      DTIME_OLD = ARRAY(5)
C
C Estimate strain rate.
      IF (TIME(2).NE.0.0) THEN
         IF (DTIME_OLD.GT.0.0) THEN
            STRAIN_RATE = (EPS-EPS_OLD) / DTIME_OLD
         ELSE
            STRAIN_RATE = SR_OLD
         END IF
      ELSE
         STRAIN_RATE = 0.0005
      END IF
C
C Limit strain rate to avoid zero/negative values and too large values.
      SR = MAX(STRAIN_RATE,1.0E-5)
      SR = MIN(SR,1.0)
C
      EPS0 = EPS
C
C ----------------------------------------------------------------------
C Polynomial coefficients.
C Here coefficients are interpreted in the same order as Python:
C     COEFFS_K = [a0, a1, a2]  ->  K = a0 + a1*eps + a2*eps**2
C     COEFFS_M = [b0, b1, b2]  ->  m = b0 + b1*eps + b2*eps**2
C
C If your Python code used numpy.polyval(COEFFS, eps), replace the two
C formulas below by:
C     K_VAL = 264.455613254*EPS**2 -177.37322503*EPS
C    1        +27.0304578037
C     M_VAL = 0.41365193633*EPS**2 -0.263045279559*EPS
C    1        +0.128785905521
C ----------------------------------------------------------------------
      K_VAL = 264.455613254 - 177.37322503*EPS
     1        + 27.0304578037*EPS**2
C
      M_VAL = 0.41365193633 - 0.263045279559*EPS
     1        + 0.128785905521*EPS**2
C
C Safety limits: K and m must be positive.
      K_MIN = 1.0E-6
      M_MIN = 1.0E-6
      K_VAL = MAX(K_VAL,K_MIN)
      M_VAL = MAX(M_VAL,M_MIN)
C
C Convert sigma = K*SR**m to Abaqus form epsdot = PLM*sigma**ESO.
      ESO = 1.0 / M_VAL
      PLM = K_VAL ** (-ESO)
C
C Define dependencies (FIELD)
      FIELD(1) = PLM
      FIELD(2) = ESO
      FIELD(3) = EPS0
      FIELD(4) = STRAIN_RATE
      FIELD(5) = DTIME
      FIELD(6) = K_VAL
      FIELD(7) = M_VAL
C
C Store values in state variables
      STATEV(1) = PLM
      STATEV(2) = ESO
      STATEV(3) = EPS0
      STATEV(4) = STRAIN_RATE
      STATEV(5) = DTIME
      STATEV(6) = K_VAL
      STATEV(7) = M_VAL
C
      IF (JRCD.NE.0) THEN
         WRITE(6,*) 'REQUEST ERROR IN USDFLD FOR ELEMENT NUMBER ',
     1        NOEL,' INTEGRATION POINT NUMBER ',NPT
      END IF
C
      RETURN
      END
