      MODULE commondata
        DOUBLE PRECISION maxSR
      END MODULE commondata


      SUBROUTINE UAMP(
     *     ampName, time, ampValueOld, dt, nProps, props, nSvars,
     *     svars, lFlagsInfo,
     *     nSensor, sensorValues, sensorNames, jSensorLookUpTable,
     *     AmpValueNew,
     *     lFlagsDefine,
     *     AmpDerivative, AmpSecDerivative, AmpIncIntegral,
     *     AmpDoubleIntegral)

      USE commondata
      INCLUDE 'ABA_PARAM.INC'

C ---------------------------------------------------------------------
C     Abaqus UAMP standard declarations
C ---------------------------------------------------------------------

      PARAMETER (iStepTime        = 1,
     *           iTotalTime       = 2,
     *           nTime            = 2)

      PARAMETER (iInitialization   = 1,
     *           iRegularInc       = 2,
     *           iCuts             = 3,
     *           ikStep            = 4,
     *           nFlagsInfo        = 4)

      PARAMETER (iComputeDeriv       = 1,
     *           iComputeSecDeriv    = 2,
     *           iComputeInteg       = 3,
     *           iComputeDoubleInteg = 4,
     *           iStopAnalysis       = 5,
     *           iConcludeStep       = 6,
     *           nFlagsDefine        = 6)

      DIMENSION time(nTime), lFlagsInfo(nFlagsInfo),
     *          lFlagsDefine(nFlagsDefine)
      DIMENSION jSensorLookUpTable(*)
      DIMENSION sensorValues(nSensor), svars(nSvars), props(nProps)

      CHARACTER*80 sensorNames(nSensor)
      CHARACTER*80 ampName

      DOUBLE PRECISION SR_TRG, SR_CUR, SR_MIN_NUM
      DOUBLE PRECISION P0, T_RAMP
      DOUBLE PRECISION UPDATE_TIME, NEXT_TIME
      DOUBLE PRECISION KP, KI, KD
      DOUBLE PRECISION ERR, ERR_OLD, INT_ERR, DERR
      DOUBLE PRECISION DT_CTRL
      DOUBLE PRECISION P_TRIAL, DP, DPMAX
      DOUBLE PRECISION P_MAX, P_MIN

C ---------------------------------------------------------------------
C     Target strain rate for Smirnov equation at 800 C:
C
C       SR_OPT = (sqrt(SIG0*SIGS)/KV)**(1/MV)
C
C     For:
C       SIG0 = 8.61
C       SIGS = 142.80
C       KV   = 2.702E5
C       MV   = 1.081
C
C       SR_OPT is approximately 2.54E-4 1/s
C ---------------------------------------------------------------------

      SR_TRG = 2.54D-4

C ---------------------------------------------------------------------
C     Initial pressure ramp
C
C     Important:
C     If the pressure load magnitude in Abaqus is equal to 1.0,
C     then AmpValueNew is the pressure value.
C     If the load magnitude is not 1.0, then AmpValueNew is only
C     a multiplier.
C ---------------------------------------------------------------------

      P0     = 0.01D0
      T_RAMP = 1.0D0

C ---------------------------------------------------------------------
C     Controller parameters
C ---------------------------------------------------------------------

      UPDATE_TIME = 0.05D0

C     Start with a soft controller.
C     If the pressure changes too slowly, increase KP later.
      KP = 0.05D0
      KI = 0.00D0
      KD = 0.00D0

C     Maximum pressure-rate limit
      DPMAX = 0.005D0

C     Absolute pressure bounds
      P_MIN = 0.0D0
      P_MAX = 1.5D0

C     Numerical protection against log(0)
      SR_MIN_NUM = 1.0D-12

C ---------------------------------------------------------------------
C     Initialization
C ---------------------------------------------------------------------

      IF (lFlagsInfo(iInitialization) .EQ. 1) THEN
         maxSR = 0.0D0

         IF (nSvars .GE. 1) svars(1) = 0.0D0
         IF (nSvars .GE. 2) svars(2) = 0.0D0
         IF (nSvars .GE. 3) svars(3) = 0.0D0
         IF (nSvars .GE. 4) svars(4) = 0.0D0
      END IF

C ---------------------------------------------------------------------
C     Stop analysis if pressure reaches the upper limit
C ---------------------------------------------------------------------

      IF (ampValueOld .GT. P_MAX) THEN
         lFlagsDefine(iStopAnalysis) = 1
      END IF

C ---------------------------------------------------------------------
C     Initial smooth pressure growth
C ---------------------------------------------------------------------

      IF (time(iStepTime) .LT. T_RAMP) THEN

         AmpValueNew = P0 * time(iStepTime) / T_RAMP

      ELSE

         AmpValueNew = ampValueOld

         NEXT_TIME = svars(1) + UPDATE_TIME

         IF (time(iStepTime) .GE. NEXT_TIME) THEN

C ------------------------------------------------------------------
C           Current maximum strain rate collected in USDFLD
C ------------------------------------------------------------------

            SR_CUR = DMAX1(maxSR, SR_MIN_NUM)

C ------------------------------------------------------------------
C           Logarithmic error:
C
C           ERR > 0 : current strain rate is too small,
C                     pressure should increase.
C
C           ERR < 0 : current strain rate is too large,
C                     pressure should decrease.
C ------------------------------------------------------------------

            ERR = DLOG(SR_TRG / SR_CUR)

            DT_CTRL = time(iStepTime) - svars(1)
            IF (DT_CTRL .LE. 1.0D-12) DT_CTRL = UPDATE_TIME

            ERR_OLD = svars(2)
            INT_ERR = svars(3)

            INT_ERR = INT_ERR + ERR * DT_CTRL
            DERR = (ERR - ERR_OLD) / DT_CTRL

C ------------------------------------------------------------------
C           Multiplicative pressure update
C ------------------------------------------------------------------

            P_TRIAL = ampValueOld *
     1                DEXP(KP * ERR + KI * INT_ERR + KD * DERR)

C ------------------------------------------------------------------
C           Pressure-rate limiting
C ------------------------------------------------------------------

            DP = P_TRIAL - ampValueOld

            IF (DP .GT. DPMAX * DT_CTRL) THEN
               DP = DPMAX * DT_CTRL
            END IF

            IF (DP .LT. -DPMAX * DT_CTRL) THEN
               DP = -DPMAX * DT_CTRL
            END IF

            AmpValueNew = ampValueOld + DP

C ------------------------------------------------------------------
C           Absolute pressure bounds
C ------------------------------------------------------------------

            IF (AmpValueNew .GT. P_MAX) AmpValueNew = P_MAX
            IF (AmpValueNew .LT. P_MIN) AmpValueNew = P_MIN

C ------------------------------------------------------------------
C           Save controller state variables
C ------------------------------------------------------------------

            svars(1) = time(iStepTime)
            svars(2) = ERR
            svars(3) = INT_ERR
            svars(4) = SR_CUR

C ------------------------------------------------------------------
C           Reset maximum strain rate for the next control window
C ------------------------------------------------------------------

            maxSR = 0.0D0

         END IF

      END IF

      RETURN
      END


      SUBROUTINE USDFLD(FIELD,STATEV,PNEWDT,DIRECT,T,CELENT,
     1 TIME,DTIME,CMNAME,ORNAME,NFIELD,NSTATV,NOEL,NPT,LAYER,
     2 KSPT,KSTEP,KINC,NDI,NSHR,COORD,JMAC,JMATYP,MATLAYO,
     3 LACCFLA)

      USE commondata
      INCLUDE 'ABA_PARAM.INC'

      CHARACTER*80 CMNAME,ORNAME
      CHARACTER*3  FLGRAY(15)

      DIMENSION FIELD(NFIELD),STATEV(NSTATV),DIRECT(3,3),
     1 T(3,3),TIME(2)

      DIMENSION ARRAY(15),JARRAY(15),JMAC(*),JMATYP(*),
     1 COORD(*)

      DOUBLE PRECISION SIG0, SIGS, KV, MV
      DOUBLE PRECISION EPS, EPS_OLD, DTIME_OLD
      DOUBLE PRECISION SR_RAW, SR_OLD
      DOUBLE PRECISION SR_OPT, SR_MODEL
      DOUBLE PRECISION SR_MIN_M, SR_MAX_M
      DOUBLE PRECISION XSM, SIGMA, MLOC, ESO, PLM
      DOUBLE PRECISION XX, YY, CNTRL
      DOUBLE PRECISION SMALL, SR_FLOOR
      INTEGER JRCD

C ---------------------------------------------------------------------
C     Smirnov equation parameters at 800 C
C ---------------------------------------------------------------------

      PARAMETER (SIG0 = 8.61D0)
      PARAMETER (SIGS = 142.80D0)
      PARAMETER (KV   = 2.702D5)
      PARAMETER (MV   = 1.081D0)

      PARAMETER (SMALL    = 1.0D-30)
      PARAMETER (SR_FLOOR = 1.0D-12)

C ---------------------------------------------------------------------
C     Optimal strain rate for Smirnov equation:
C
C       SR_OPT = (sqrt(SIG0*SIGS)/KV)**(1/MV)
C ---------------------------------------------------------------------

      SR_OPT = (DSQRT(SIG0 * SIGS) / KV)**(1.0D0 / MV)

C ---------------------------------------------------------------------
C     Interval for material-property calculation.
C
C     Important:
C     This interval is not the pressure-control target.
C     It is used only to avoid unrealistic PLM/ESO when SR_RAW is
C     extremely small or extremely large.
C
C     Start with a narrow interval around SR_OPT.
C     Later it can be expanded.
C ---------------------------------------------------------------------

      SR_MIN_M = 2.0D-4
      SR_MAX_M = 3.2D-4

C ---------------------------------------------------------------------
C     Current equivalent creep strain
C ---------------------------------------------------------------------

      CALL GETVRM('CE',ARRAY,JARRAY,FLGRAY,JRCD,JMAC,JMATYP,
     1 MATLAYO,LACCFLA)

      EPS = DMAX1(DBLE(ARRAY(7)), 0.0D0)

C ---------------------------------------------------------------------
C     Old state variables
C
C     STATEV(3) = previous equivalent creep strain
C     STATEV(4) = previous raw strain rate
C     STATEV(5) = previous time increment
C     STATEV(7) = control-zone flag
C ---------------------------------------------------------------------

      CALL GETVRM('SDV',ARRAY,JARRAY,FLGRAY,JRCD,JMAC,JMATYP,
     1 MATLAYO,LACCFLA)

      EPS_OLD   = DBLE(ARRAY(3))
      SR_OLD    = DBLE(ARRAY(4))
      DTIME_OLD = DBLE(ARRAY(5))
      CNTRL     = DBLE(ARRAY(7))

C ---------------------------------------------------------------------
C     Calculate actual strain rate from strain increment.
C
C     SR_RAW is used for pressure control.
C ---------------------------------------------------------------------

      IF (TIME(2) .GT. 0.0D0 .AND. DTIME_OLD .GT. SMALL) THEN
         SR_RAW = (EPS - EPS_OLD) / DTIME_OLD
      ELSE
         SR_RAW = SR_OPT
      END IF

C ---------------------------------------------------------------------
C     Protection against negative or zero strain rate.
C ---------------------------------------------------------------------

      IF (SR_RAW .LE. SR_FLOOR) THEN
         IF (SR_OLD .GT. SR_FLOOR) THEN
            SR_RAW = SR_OLD
         ELSE
            SR_RAW = SR_OPT
         END IF
      END IF

C ---------------------------------------------------------------------
C     SR_MODEL is used only for material-property calculation.
C
C     Do not use unrestricted SR_RAW for PLM/ESO, because for very small
C     strain rates the Smirnov equation gives m -> 0 and ESO -> infinity.
C ---------------------------------------------------------------------

      SR_MODEL = DMIN1(SR_MAX_M, DMAX1(SR_RAW, SR_MIN_M))

C ---------------------------------------------------------------------
C     Smirnov equation:
C
C       sigma = sigs * (sig0 + kv * SR^mv)
C                    / (sigs + kv * SR^mv)
C ---------------------------------------------------------------------

      XSM = KV * SR_MODEL**MV

      SIGMA = SIGS * (SIG0 + XSM) / (SIGS + XSM)

C ---------------------------------------------------------------------
C     Local strain-rate sensitivity:
C
C       m = d ln(sigma) / d ln(SR)
C ---------------------------------------------------------------------

      MLOC = MV * XSM * (SIGS - SIG0) /
     1       ((SIG0 + XSM) * (SIGS + XSM))

C     Additional protection against unrealistic creep exponent
      IF (MLOC .LT. 0.30D0) MLOC = 0.30D0

C ---------------------------------------------------------------------
C     Abaqus creep form:
C
C       SR = PLM * sigma^ESO
C
C       ESO = 1 / m
C       PLM = SR / sigma^ESO
C ---------------------------------------------------------------------

      ESO = 1.0D0 / MLOC

      IF (SIGMA .GT. SMALL) THEN
         PLM = DEXP(DLOG(SR_MODEL) - ESO * DLOG(SIGMA))
      ELSE
         PLM = 0.0D0
      END IF

C ---------------------------------------------------------------------
C     Select control zone.
C
C     This is the same coordinate-based zone as in your old script.
C     Change this condition if you need to control another part
C     of the blank.
C ---------------------------------------------------------------------

      XX = COORD(1)
      YY = COORD(2)

      IF (CNTRL .LT. 0.5D0) THEN

         IF (YY .LT. 200.6D0 .AND. YY .GT. 0.0D0 .AND.
     1       XX .LT. 5.125D0 .AND. XX .GT. 0.0D0) THEN
            CNTRL = 2.0D0
         ELSE
            CNTRL = 1.0D0
         END IF

      END IF

C ---------------------------------------------------------------------
C     Collect maximum actual strain rate in the control zone.
C
C     Important:
C     The pressure controller must use SR_RAW, not SR_MODEL.
C ---------------------------------------------------------------------

      IF (CNTRL .GT. 1.5D0) THEN
         IF (SR_RAW .GT. maxSR) maxSR = SR_RAW
      END IF

C ---------------------------------------------------------------------
C     Field variables
C
C     FIELD(1) = PLM
C     FIELD(2) = ESO
C     FIELD(3) = equivalent creep strain
C     FIELD(4) = actual raw strain rate
C     FIELD(5) = current time increment
C     FIELD(6) = maximum strain rate in control zone
C     FIELD(7) = control-zone flag
C     FIELD(8) = local m
C     FIELD(9) = Smirnov stress
C     FIELD(10) = strain rate used for PLM/ESO
C ---------------------------------------------------------------------

      IF (NFIELD .GE. 1) FIELD(1) = PLM
      IF (NFIELD .GE. 2) FIELD(2) = ESO
      IF (NFIELD .GE. 3) FIELD(3) = EPS
      IF (NFIELD .GE. 4) FIELD(4) = SR_RAW
      IF (NFIELD .GE. 5) FIELD(5) = DTIME
      IF (NFIELD .GE. 6) FIELD(6) = maxSR
      IF (NFIELD .GE. 7) FIELD(7) = CNTRL
      IF (NFIELD .GE. 8) FIELD(8) = MLOC
      IF (NFIELD .GE. 9) FIELD(9) = SIGMA
      IF (NFIELD .GE. 10) FIELD(10) = SR_MODEL

C ---------------------------------------------------------------------
C     State variables
C
C     STATEV(1) = PLM
C     STATEV(2) = ESO
C     STATEV(3) = equivalent creep strain
C     STATEV(4) = actual raw strain rate
C     STATEV(5) = current time increment
C     STATEV(6) = maximum strain rate in control zone
C     STATEV(7) = control-zone flag
C     STATEV(8) = local m
C     STATEV(9) = Smirnov stress
C     STATEV(10) = strain rate used for PLM/ESO
C ---------------------------------------------------------------------

      IF (NSTATV .GE. 1) STATEV(1) = PLM
      IF (NSTATV .GE. 2) STATEV(2) = ESO
      IF (NSTATV .GE. 3) STATEV(3) = EPS
      IF (NSTATV .GE. 4) STATEV(4) = SR_RAW
      IF (NSTATV .GE. 5) STATEV(5) = DTIME
      IF (NSTATV .GE. 6) STATEV(6) = maxSR
      IF (NSTATV .GE. 7) STATEV(7) = CNTRL
      IF (NSTATV .GE. 8) STATEV(8) = MLOC
      IF (NSTATV .GE. 9) STATEV(9) = SIGMA
      IF (NSTATV .GE. 10) STATEV(10) = SR_MODEL

      IF (JRCD .NE. 0) THEN
         WRITE(6,*) 'REQUEST ERROR IN USDFLD FOR ELEMENT ',NOEL,
     1              ' INTEGRATION POINT ',NPT
      END IF

      RETURN
      END