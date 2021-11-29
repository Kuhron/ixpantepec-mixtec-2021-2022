# script to get fundamental frequency values from vowels that we have manually segmented

# resources
# https://kbmcgowan.github.io/teaching/Praat/Stanford-Day2.pdf
# https://kbmcgowan.github.io/teaching/Praat/Stanford-Day3.pdf
# https://www.fon.hum.uva.nl/praat/manual/
# https://praatscripting.lingphon.net/loops-2.html
# https://github.com/FieldDB/Praat-Scripts/blob/master/get.5spectral.peak.all.intervals.v14.praat
# https://www.fon.hum.uva.nl/praat/manual/Script_for_listing_F0_statistics.html

# select the Sound and TextGrid objects
selectObject: 1, 2

# selected$ uses the string name of the object, just sound uses a unique identifier which is better
soundId = selected ("Sound")
soundName$ = selected$ ("Sound")
tgId = selected ("TextGrid")
tgName$ = selected$ ("TextGrid")

writeInfoLine: "start"
appendInfoLine: soundId, " = ", soundName$
appendInfoLine: tgId , " = ", tgName$

# create a Pitch object for F0 analysis
# first check if it already exists
# https://groups.io/g/Praat-Users-List/topic/scripting_how_to_check_if_an/71951441
# if you just do selectObject without nocheck, it will error when no such object exists
nocheck selectObject: "Pitch " + soundName$
if numberOfSelected ("Pitch") > 0
    appendInfoLine: "Pitch object already exists"
    pitchId = selected ("Pitch")
else
    appendInfoLine: "Pitch object not found, creating a new one"
    select soundId
    
    # create a Pitch object; note there are various algorithms for this, need to see what works

    # default method
    # To Pitch... kwargs: timeStepSecs pitchFloorHz
    # To Pitch... 0.01 75 600

    # cross-correlation method
    To Pitch (cc)... 0.0 75 15 false 0.03 0.45 0.01 0.35 0.14 2000

    # shs method
    # https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__shs____.html; NOTE that the args are in the WRONG ORDER on the documentation! use the order from the actual To Pitch (shs) dialog in the Praat program itself
    # To Pitch (shs)... 0.01 100 15 1250 15 0.84 500 48

    selectObject: "Pitch"
    pitchId = selected ("Pitch")
endif
appendInfoLine: "Pitch object ID is ", pitchId


select tgId
nTiers = Get number of tiers
appendInfoLine: nTiers, " tiers"

# which tier has the vowel segments?
desiredTierName$ = "vowel"
for tierNumber from 1 to nTiers
    tierName$ = Get tier name: tierNumber
    appendInfoLine: "Tier number ", tierNumber, " is named ", tierName$
    if tierName$ = desiredTierName$
        vowelTierNumber = tierNumber
        # don't know how to break in Praat
    endif
endfor
appendInfoLine: "The vowel tier has number ", vowelTierNumber

# get the labeled intervals
nIntervals = Get number of intervals: vowelTierNumber

# you don't have to initialize arrays and dicts, you can just start assigning to an undeclared one, weird
# intervalNumberToName$ = []
# intervalNumberToStart = []
# intervalNumberToEnd = []

for intervalNumber from 1 to nIntervals
    intervalName$ = Get label of interval: vowelTierNumber, intervalNumber
    intervalNumberToName$ [intervalNumber] = intervalName$
    start = Get start point: vowelTierNumber, intervalNumber
    end = Get end point: vowelTierNumber, intervalNumber
    intervalNumberToStart [intervalNumber] = start
    intervalNumberToEnd [intervalNumber] = end
endfor

select pitchId
for intervalNumber from 1 to nIntervals
    intervalName$ = intervalNumberToName$ [intervalNumber]
    if intervalName$ <> ""
        start = intervalNumberToStart [intervalNumber]
        end = intervalNumberToEnd [intervalNumber]
        appendInfoLine: "interval #", intervalNumber, tab$, "label: ", intervalName$, tab$, "start: ", start, tab$, "end: ", end
        duration = end - start
        timeStep = 0.025
        nTimeSteps = floor(duration / timeStep)
        for step from 1 to nTimeSteps
            tmin = start + (step - 1) * timeStep
            tmax = tmin + timeStep
            fMean = Get mean: tmin, tmax, "Hertz"
            fMin = Get minimum: tmin, tmax, "Hertz", "Parabolic"
            fMax = Get maximum: tmin, tmax, "Hertz", "Parabolic"
            stdev = Get standard deviation: tmin, tmax, "Hertz"
            appendInfoLine: "step #", step, "/", nTimeSteps, tab$, "fMin: ", fMin, tab$, "fMax: ", fMax, tab$, "fMean: ", fMean, tab$, "stdev: ", stdev
        endfor
    endif
endfor

appendInfoLine: "done"
