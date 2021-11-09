# script to get fundamental frequency values from vowels that we have manually segmented

# resources
# https://kbmcgowan.github.io/teaching/Praat/Stanford-Day2.pdf
# https://kbmcgowan.github.io/teaching/Praat/Stanford-Day3.pdf
# https://www.fon.hum.uva.nl/praat/manual/
# https://praatscripting.lingphon.net/loops-2.html
# https://github.com/FieldDB/Praat-Scripts/blob/master/get.5spectral.peak.all.intervals.v14.praat

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
for intervalNumber from 1 to nIntervals
	intervalName$ = Get label of interval: vowelTierNumber, intervalNumber
	if intervalName$ <> ""
		start = Get start point: vowelTierNumber, intervalNumber
		end = Get end point: vowelTierNumber, intervalNumber
		appendInfoLine: intervalNumber, tab$, intervalName$, tab$, start, tab$, end
	endif
endfor

appendInfoLine: "done"
