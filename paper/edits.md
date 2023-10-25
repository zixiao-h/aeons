## General comments:

1) You should expand the citation list. Buchner's review is an excellent source (and should also be cited alongside Ashton):

https://arxiv.org/abs/2101.09675

In particular you should cite anything which looks relevant from the 'Theory' section in 4.

2) The abstract and conclusion should also mention that we have a detailed discussion of the anatomy of nested sampling, and present new analytic results for understanding this.

4) 4.2 should have a few sentences indicating that whilst a more complete Bayesian approach might be possible, as we wish this to be done at run-time we opt for a more pragmatic method for estimating the end point.
<!-- 3) is it possible to put the dead points onto Figures 3 & 4? This would be a horizontal line in the former, and something power-law/exponential in the latter, continuously connected to the peak of the live point distribution -->

## More minor suggestions:

- Mention Skilling's paper when discussing uncertainties in evidence at end of section 2
- When you make the statement 'the remaining live points are then killed off one-by-one', could you emphasise that this is quantitatively equivalent to the phrasing of the existing literature (which treats the evidence in the final live points separately), but qualitatively neater (cite the dynesty paper here).
- Is it possible to have some line pairing the axvlines in the lower panel with the the corresponding upper panels (in analogy with Figure 3)
<!-- - In Figure 2, can we colour the successive live points in different colours (as in the original plot) -->
<!-- - eq (22) -- can we label this something other than $\beta^*$ -- I would rather call the Habeck temperature $\beta^*$, since in the end this is our concluded best temperature -->
<!-- - "Define a likelihood normalised": sentence fragment -->
<!-- - Section 3: don't begin paragraph with Hence -->
<!-- - unique for estimating -> unique in its estimation of -->
<!-- - At equation (8) you should mention the connection to equation (6) -->
<!-- - In most instances where you say nlive, I would switch this to n_i, to emphasise that these methods are just as valid for dynamic nested sampling -->
<!-- - I like your arguments at the end of section 3.2. These should be in display equations rather than inline. -->
<!-- - (27) could we refer to T{impl} as f_\mathrm{sampler}: -->
<!-- https://github.com/williamjameshandley/talks/raw/kcl_2023/will_handley_kcl_2023.pdf -->
<!-- it was rightly pointed out to me by Torsten Ensslin that (27) is not dimensionally consistent! You can also cite Will Handley "The scaling frontier of nested sampling" (Nov 2023) in prep alongside that talk. -->

## Other
I didn't spot many typos, but ideally you should run this through LanguageTool (overleaf plugin) or equivalent

After you've made the above adjustments, you should pass a copy by Artyom <ab399001@gmail.com>, in particular checking the spelling of his name (he uses multiple spellings, so should choose the one he wants to be 'permanent' were he to pursue an academic career -- quote me on this.

