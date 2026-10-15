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
C     User material law:
C         SIGMA = K(EPS) * SR ** M(EPS)
C
C     Abaqus creep power-law form used through field variables:
C         SR = PLM * SIGMA ** ESO
C     Therefore:
C         ESO = 1 / M(EPS)
C         PLM = K(EPS) ** (-1 / M(EPS))
C
      REAL STRAIN_RATE, SR, SR_MIN, SR_MAX
      REAL EPS, EPS0, EPS_OLD, EPS_USED, EPS_MAX, D_EPS, EPSMIN
      REAL KPOLY, MPOLY, STRESS, ESO, PLM
      REAL K_MIN, M_MIN, M_MAX
      REAL SR_OLD, DTIME_OLD
C
C     Parameters for numerical safety / extrapolation control
      PARAMETER (SR_MIN = 1.0E-6)
      PARAMETER (SR_MAX = 1.0E+0)
      PARAMETER (EPS_MAX = 1.5E+0)
      PARAMETER (D_EPS  = 1.0E-1)
      PARAMETER (K_MIN  = 1.0E-6)
      PARAMETER (M_MIN  = 1.0E-3)
      PARAMETER (M_MAX  = 1.0E+1)
C
C     Get current equivalent creep strain CE.
C     In Abaqus CE components usually contain:
C       ARRAY(7) = equivalent creep strain CEeq
      CALL GETVRM('CE',ARRAY,JARRAY,FLGRAY,JRCD,JMAC,JMATYP,
     1 MATLAYO,LACCFLA)
      EPS = ARRAY(7)
C
C     Get old state variables from previous increment.
C       STATEV(3) = previous equivalent creep strain
C       STATEV(4) = previous estimated strain rate
C       STATEV(5) = previous DTIME
      CALL GETVRM('SDV',ARRAY,JARRAY,FLGRAY,JRCD,JMAC,JMATYP,
     1 MATLAYO,LACCFLA)
      EPS_OLD   = ARRAY(3)
      SR_OLD    = ARRAY(4)
      DTIME_OLD = ARRAY(5)
C
C     Estimate current equivalent strain rate from CE increment.
C     On the first call use a small initial value.
      IF (TIME(2).NE.0.0 .AND. DTIME_OLD.GT.0.0) THEN
         STRAIN_RATE = (EPS - EPS_OLD) / DTIME_OLD
      ELSE
         STRAIN_RATE = 5.0E-4
      END IF
C
C     Protect against negative/small values and too large values.
      SR = MAX(STRAIN_RATE, SR_MIN)
      SR = MIN(SR, SR_MAX)
C
C     Save original strain for output, then smoothly limit EPS used
C     in polynomials to avoid strong extrapolation after EPS_MAX.
      EPS0 = EPS
      EPSMIN = MIN(EPS_MAX, EPS)
      EPS_USED = EPSMIN + D_EPS - D_EPS*EXP((EPSMIN-EPS)/D_EPS)
C
C     Polynomial coefficients are used as:
C       K(EPS) = 264.455613254*EPS**2
C              - 177.37322503*EPS + 27.0304578037
C       M(EPS) = 0.41365193633*EPS**2
C              - 0.263045279559*EPS + 0.128785905521
C
      KPOLY = 0.0
      KPOLY = KPOLY + 264.455613254*EPS_USED**2
      KPOLY = KPOLY - 177.37322503*EPS_USED
      KPOLY = KPOLY + 27.0304578037
C
      MPOLY = 0.0
      MPOLY = MPOLY + 0.41365193633*EPS_USED**2
      MPOLY = MPOLY - 0.263045279559*EPS_USED
      MPOLY = MPOLY + 0.128785905521
C
C     Numerical protection. K must be positive, m must be positive.
      KPOLY = MAX(KPOLY, K_MIN)
      MPOLY = MAX(MPOLY, M_MIN)
      MPOLY = MIN(MPOLY, M_MAX)
C
C     Convert sigma = K*SR**m into Abaqus power-law creep form:
C       SR = PLM*SIGMA**ESO
      ESO = 1.0 / MPOLY
      PLM = KPOLY ** (-ESO)
C
C     Diagnostic stress corresponding to the estimated current SR.
      STRESS = KPOLY * SR ** MPOLY
C
C     Define field variables / dependencies.
C       FIELD(1): PLM = A
C       FIELD(2): ESO = n
C       FIELD(3): original equivalent creep strain
C       FIELD(4): estimated equivalent strain rate
C       FIELD(5): current DTIME
C       FIELD(6): K(EPS), if NFIELD >= 6
C       FIELD(7): m(EPS), if NFIELD >= 7
C       FIELD(8): diagnostic stress, if NFIELD >= 8
      FIELD(1) = PLM
      FIELD(2) = ESO
      FIELD(3) = EPS0
      FIELD(4) = STRAIN_RATE
      FIELD(5) = DTIME
      IF (NFIELD.GE.6) FIELD(6) = KPOLY
      IF (NFIELD.GE.7) FIELD(7) = MPOLY
      IF (NFIELD.GE.8) FIELD(8) = STRESS
C
C     Store state variables for the next increment.
      STATEV(1) = PLM
      STATEV(2) = ESO
      STATEV(3) = EPS0
      STATEV(4) = STRAIN_RATE
      STATEV(5) = DTIME
      IF (NSTATV.GE.6) STATEV(6) = KPOLY
      IF (NSTATV.GE.7) STATEV(7) = MPOLY
      IF (NSTATV.GE.8) STATEV(8) = STRESS
C
      IF (JRCD.NE.0) THEN
         WRITE(6,*) 'REQUEST ERROR IN USDFLD FOR ELEMENT NUMBER ',
     1      NOEL,' INTEGRATION POINT NUMBER ',NPT
      END IF
C
      RETURN
      END
