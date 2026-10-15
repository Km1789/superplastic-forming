      SUBROUTINE USDFLD(FIELD,STATEV,PNEWDT,DIRECT,T,CELENT,
     1 TIME,DTIME,CMNAME,ORNAME,NFIELD,NSTATV,NOEL,NPT,LAYER,
     2 KSPT,KSTEP,KINC,NDI,NSHR,COORD,JMAC,JMATYP,MATLAYO,
     3 LACCFLA)

      INCLUDE 'ABA_PARAM.INC'

      CHARACTER*80 CMNAME,ORNAME
      CHARACTER*3  FLGRAY(15)

      DIMENSION FIELD(NFIELD),STATEV(NSTATV),DIRECT(3,3),
     1 T(3,3),TIME(2)
      DIMENSION ARRAY(15),JARRAY(15),JMAC(*),JMATYP(*),
     1 COORD(*)

C ---------------------------------------------------------------------
C     Smirnov equation at T = 800 C:
C
C        sigma = sigs * (sig0 + kv * SR**mv)
C                      / (sigs + kv * SR**mv)
C
C     The creep law used by Abaqus is approximated locally as:
C
C        SR = PLM * sigma**ESO
C
C     where
C
C        ESO = d ln(SR) / d ln(sigma) = 1 / m(SR)
C        PLM = SR / sigma**ESO
C ---------------------------------------------------------------------

      DOUBLE PRECISION SIG0, SIGS, KV, MV
      DOUBLE PRECISION EPS, EPS_OLD
      DOUBLE PRECISION SR_RAW, SR, SR_OLD
      DOUBLE PRECISION SR_OPT, SR_MIN, SR_MAX
      DOUBLE PRECISION DTIME_OLD
      DOUBLE PRECISION X, STRESS, MLOC
      DOUBLE PRECISION ESO, PLM
      DOUBLE PRECISION BVAL, DISC, M_LIMIT
      DOUBLE PRECISION SMALL
      INTEGER JRCD
      INTEGER USE_CONST_OPT

C ---------------------------------------------------------------------
C     Material constants for 800 C
C ---------------------------------------------------------------------

      PARAMETER (SIG0 = 8.61D0)
      PARAMETER (SIGS = 142.80D0)
      PARAMETER (KV   = 2.702D5)
      PARAMETER (MV   = 1.081D0)

C ---------------------------------------------------------------------
C     Numerical parameters
C ---------------------------------------------------------------------

      PARAMETER (SMALL = 1.0D-30)

C ---------------------------------------------------------------------
C     USE_CONST_OPT:
C
C     1 - use constant optimal strain rate.
C         This is the first stable check recommended in the discussion.
C
C     0 - calculate strain rate from strain increment and apply MIN/MAX
C         clipping around the optimal strain-rate range.
C
C     Recommended workflow:
C       first run with USE_CONST_OPT = 1,
C       then switch to USE_CONST_OPT = 0.
C ---------------------------------------------------------------------

      PARAMETER (USE_CONST_OPT = 0)

C ---------------------------------------------------------------------
C     M_LIMIT defines the strain-rate interval by m(SR) >= M_LIMIT.
C
C     For the wide interval from the paper, use M_LIMIT = 0.50D0.
C     For a narrower and more stable interval around SR_OPT,
C     try 0.60D0, 0.63D0, 0.64D0.
C
C     At 800 C:
C       SR_OPT is approximately 2.54E-4 1/s.
C ---------------------------------------------------------------------

      PARAMETER (M_LIMIT = 0.50D0)

C ---------------------------------------------------------------------
C     Read current equivalent creep strain
C     ARRAY(7) is used as in the previous hyperbolic-sine script.
C ---------------------------------------------------------------------

      CALL GETVRM('CE',ARRAY,JARRAY,FLGRAY,JRCD,JMAC,JMATYP,
     1 MATLAYO,LACCFLA)

      EPS = DBLE(ARRAY(7))

C ---------------------------------------------------------------------
C     Read old state variables
C
C     STATEV(3) = old equivalent creep strain
C     STATEV(4) = old calculated strain rate
C     STATEV(5) = old time increment
C ---------------------------------------------------------------------

      CALL GETVRM('SDV',ARRAY,JARRAY,FLGRAY,JRCD,JMAC,JMATYP,
     1 MATLAYO,LACCFLA)

      EPS_OLD   = DBLE(ARRAY(3))
      SR_OLD    = DBLE(ARRAY(4))
      DTIME_OLD = DBLE(ARRAY(5))

C ---------------------------------------------------------------------
C     Optimal strain rate:
C
C        SR_OPT = ( sqrt(sig0 * sigs) / kv )**(1/mv)
C ---------------------------------------------------------------------

      SR_OPT = (DSQRT(SIG0 * SIGS) / KV)**(1.0D0 / MV)

C ---------------------------------------------------------------------
C     Strain-rate limits for m(SR) >= M_LIMIT.
C
C     General equation:
C
C       m(SR) = MV * X * (SIGS - SIG0)
C             / ((SIG0 + X) * (SIGS + X)),
C
C       X = KV * SR**MV.
C
C     Solving m(SR) = M_LIMIT gives:
C
C       X1,2 = (B +- sqrt(B**2 - 4*SIG0*SIGS)) / 2
C
C       B = MV/M_LIMIT * (SIGS - SIG0) - SIG0 - SIGS
C
C       SR1,2 = (X1,2 / KV)**(1/MV)
C ---------------------------------------------------------------------

      BVAL = (MV / M_LIMIT) * (SIGS - SIG0) - SIG0 - SIGS
      DISC = BVAL * BVAL - 4.0D0 * SIG0 * SIGS

      IF (DISC .GT. 0.0D0) THEN
         SR_MIN = ((BVAL - DSQRT(DISC)) /
     1            (2.0D0 * KV))**(1.0D0 / MV)

         SR_MAX = ((BVAL + DSQRT(DISC)) /
     1            (2.0D0 * KV))**(1.0D0 / MV)
      ELSE
C        Fallback: very narrow interval around SR_OPT
         SR_MIN = 0.99D0 * SR_OPT
         SR_MAX = 1.01D0 * SR_OPT
      END IF

C ---------------------------------------------------------------------
C     Calculate current strain rate.
C
C     Important:
C     SR_RAW is calculated from the strain increment.
C     Only after that MIN/MAX clipping is applied.
C ---------------------------------------------------------------------

      IF (TIME(2) .GT. 0.0D0 .AND. DTIME_OLD .GT. SMALL) THEN
         SR_RAW = (EPS - EPS_OLD) / DTIME_OLD
      ELSE
         SR_RAW = SR_OPT
      END IF

C ---------------------------------------------------------------------
C     Protection against negative or zero strain rate.
C     Negative values may appear due to numerical oscillations.
C ---------------------------------------------------------------------

      IF (SR_RAW .LE. SMALL) THEN
         IF (SR_OLD .GT. SMALL) THEN
            SR_RAW = SR_OLD
         ELSE
            SR_RAW = SR_OPT
         END IF
      END IF

C ---------------------------------------------------------------------
C     Select strain-rate mode.
C ---------------------------------------------------------------------

      IF (USE_CONST_OPT .EQ. 1) THEN
         SR = SR_OPT
      ELSE
         SR = DMIN1(SR_MAX, DMAX1(SR_RAW, SR_MIN))
      END IF

C ---------------------------------------------------------------------
C     Smirnov stress for selected strain rate.
C ---------------------------------------------------------------------

      X = KV * SR**MV

      STRESS = SIGS * (SIG0 + X) / (SIGS + X)

C ---------------------------------------------------------------------
C     Local strain-rate sensitivity:
C
C       m = d ln(sigma) / d ln(SR)
C
C     Abaqus power-law exponent:
C
C       ESO = d ln(SR) / d ln(sigma) = 1 / m
C ---------------------------------------------------------------------

      MLOC = MV * X * (SIGS - SIG0) /
     1       ((SIG0 + X) * (SIGS + X))

      IF (MLOC .LT. 1.0D-6) THEN
         MLOC = 1.0D-6
      END IF

      ESO = 1.0D0 / MLOC

C ---------------------------------------------------------------------
C     PLM = SR / STRESS**ESO
C
C     Logarithmic form is used for numerical stability.
C ---------------------------------------------------------------------

      IF (STRESS .GT. SMALL) THEN
         PLM = DEXP(DLOG(SR) - ESO * DLOG(STRESS))
      ELSE
         PLM = 0.0D0
      END IF

C ---------------------------------------------------------------------
C     Write field variables.
C
C     FIELD(1) = PLM
C     FIELD(2) = ESO
C     FIELD(3) = equivalent creep strain
C     FIELD(4) = raw strain rate from strain increment
C     FIELD(5) = current time increment
C     FIELD(6) = clipped/used strain rate, optional
C     FIELD(7) = Smirnov stress, optional
C     FIELD(8) = local m, optional
C ---------------------------------------------------------------------

      FIELD(1) = PLM
      FIELD(2) = ESO

      IF (NFIELD .GE. 3) FIELD(3) = EPS
      IF (NFIELD .GE. 4) FIELD(4) = SR_RAW
      IF (NFIELD .GE. 5) FIELD(5) = DTIME
      IF (NFIELD .GE. 6) FIELD(6) = SR
      IF (NFIELD .GE. 7) FIELD(7) = STRESS
      IF (NFIELD .GE. 8) FIELD(8) = MLOC

C ---------------------------------------------------------------------
C     Store state variables.
C
C     STATEV(1) = PLM
C     STATEV(2) = ESO
C     STATEV(3) = equivalent creep strain
C     STATEV(4) = raw strain rate
C     STATEV(5) = current time increment
C     STATEV(6) = clipped/used strain rate, optional
C     STATEV(7) = Smirnov stress, optional
C     STATEV(8) = local m, optional
C ---------------------------------------------------------------------

      STATEV(1) = PLM
      STATEV(2) = ESO

      IF (NSTATV .GE. 3) STATEV(3) = EPS
      IF (NSTATV .GE. 4) STATEV(4) = SR_RAW
      IF (NSTATV .GE. 5) STATEV(5) = DTIME
      IF (NSTATV .GE. 6) STATEV(6) = SR
      IF (NSTATV .GE. 7) STATEV(7) = STRESS
      IF (NSTATV .GE. 8) STATEV(8) = MLOC

C ---------------------------------------------------------------------
C     Error message for GETVRM
C ---------------------------------------------------------------------

      IF (JRCD .NE. 0) THEN
         WRITE(6,*) 'REQUEST ERROR IN USDFLD FOR ELEMENT ',NOEL,
     1              ' INTEGRATION POINT ',NPT
      END IF

      RETURN
      END