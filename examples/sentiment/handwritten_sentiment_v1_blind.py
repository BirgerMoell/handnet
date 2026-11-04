"""
# FROZEN HISTORICAL VERSION - kept so the corresponding table row can be regenerated. The current network is in handnet/.
A hand-written neural network for sentiment classification.
No training. No gradients. No labels seen. Every number below was typed by hand.

Architecture
------------
  tokens -> Embedding (V x 6)            E[word] = [pos, neg, negate, intensify, contrast, hedge]
         -> Gated recurrent cell (6+4 -> 4)   state h = [pos_acc, neg_acc, neg_gate, int_gate]
         -> Readout: Linear(4->6) + ReLU + Linear(6->1)
         -> sigmoid

The recurrent cell is an LSTM-style multiplicative gate cell; its weight matrices are
written out explicitly as numpy arrays. The "intelligence" lives in two places:
  (1) the embedding table (a sentiment lexicon with strengths), and
  (2) the gating weights that implement "not good", "very bad", "boring but ultimately moving".
"""
import numpy as np
import re

# ---------------------------------------------------------------------------
# 1. EMBEDDING TABLE  (hand-written)
#    axes: 0 pos  1 neg  2 negator  3 intensifier  4 contrast  5 hedge/diminisher
# ---------------------------------------------------------------------------
D = 6
POS = {
 # strength 3
 'masterpiece':3,'masterful':3,'brilliant':3,'superb':3,'magnificent':3,'extraordinary':3,'exhilarating':3,
 'stunning':3,'breathtaking':3,'flawless':3,'perfect':3,'gorgeous':3,'mesmerizing':3,'dazzling':3,'triumph':3,
 'wonderful':3,'wonderfully':3,'terrific':3,'marvelous':3,'marvelously':3,'exquisite':3,'glorious':3,
 'spectacular':3,'astonishing':3,'riveting':3,'unforgettable':3,'phenomenal':3,'sublime':3,'genius':3,
 'best':3,'greatest':3,'finest':3,'remarkable':3,'remarkably':3,'captivating':3,'enthralling':3,'joyous':3,
 'beautifully':3,'beautiful':3,'delightful':3,'delight':3,'delightfully':3,'irresistible':3,'rapturous':3,
 'thrilling':3,'thrilled':3,'awe':3,'awesome':3,'transcendent':3,'soaring':3,'wondrous':3,'electrifying':3,
 'must-see':3,'brilliantly':3,'powerfully':3,'ingenious':3,'stirring':3,'gem':3,'treasure':3,'richly':3,
 # strength 2
 'good':2,'great':2,'excellent':2,'fine':2,'funny':2,'hilarious':2,'fun':2,'enjoyable':2,'enjoy':2,'enjoyed':2,
 'entertaining':2,'entertains':2,'entertainment':1,'charming':2,'charm':2,'lovely':2,'love':2,'loved':2,'loves':2,
 'moving':2,'touching':2,'poignant':2,'heartfelt':2,'heartwarming':2,'warm':2,'warmth':2,'tender':2,
 'engaging':2,'engrossing':2,'absorbing':2,'compelling':2,'gripping':2,'fascinating':2,'intriguing':2,
 'smart':2,'clever':2,'witty':2,'wit':2,'sharp':2,'intelligent':2,'thoughtful':2,'thought-provoking':2,
 'powerful':2,'strong':2,'solid':2,'satisfying':2,'rewarding':2,'refreshing':2,'inspiring':2,'inspired':2,
 'original':2,'fresh':2,'inventive':2,'imaginative':2,'creative':2,'innovative':2,'vivid':2,'lively':2,
 'impressive':2,'impressed':2,'accomplished':2,'effective':2,'affecting':2,'haunting':2,'memorable':2,
 'honest':2,'sincere':2,'genuine':2,'authentic':2,'nuanced':2,'subtle':2,'elegant':2,'graceful':2,
 'hilariously':2,'deftly':2,'deft':2,'skillfully':2,'skillful':2,'assured':2,'confident':2,'polished':2,
 'sweet':2,'likable':2,'likeable':2,'appealing':2,'winning':2,'winner':2,'win':1,'wins':1,'triumphant':2,
 'success':2,'successful':2,'succeeds':2,'works':2,'worth':2,'worthy':2,'worthwhile':2,'recommend':2,
 'rich':2,'rare':2,'joy':2,'pleasure':2,'pleasant':2,'pleasing':2,'pleasurable':2,'pleases':2,'happy':2,
 'amusing':2,'amused':2,'laughs':2,'laugh':2,'laughter':2,'comic':1,'humor':1,'humorous':2,'playful':2,
 'exciting':2,'excitement':2,'thrills':2,'suspenseful':2,'tense':1,'taut':2,'crisp':2,'brisk':2,
 'beauty':2,'bold':2,'daring':2,'ambitious':2,'provocative':2,'profound':2,'insightful':2,'insight':2,
 'admirable':2,'admirably':2,'impeccable':2,'well-crafted':2,'well-made':2,'well-acted':2,'well-written':2,
 'artful':2,'artistry':2,'surprising':2,'surprisingly':1,'unexpected':1,'better':1,'nice':2,'nicely':2,
 'cool':2,'hip':1,'stylish':2,'style':1,'sexy':2,'exuberant':2,'infectious':2,'energetic':2,'energy':1,
 'uplifting':2,'hopeful':2,'hope':1,'grace':2,'dignity':2,'compassion':2,'compassionate':2,'generous':2,
 'enchanting':2,'enchanted':2,'magical':2,'magic':2,'spirited':2,'spirit':1,'soulful':2,'soul':1,
 'resonant':2,'resonates':2,'rings':1,'true':1,'truth':1,'illuminating':2,'enlightening':2,'eye-opening':2,
 'deeply':1,'deep':1,'depth':2,'complex':1,'layered':2,'textured':2,'lush':2,'luminous':2,'radiant':2,
 'cute':1,'adorable':2,'endearing':2,'heart':1,'hearts':1,'crowd-pleaser':2,'pleaser':2,'blast':2,'treat':2,
 'perfectly':2,'ideal':2,'wise':2,'wisdom':2,'mature':2,'dramatic':1,'drama':0,'strength':1,'strongest':2,
 'potent':2,'intense':1,'exhilarating':3,'invigorating':2,'bracing':2,'dynamic':2,'vibrant':2,'vital':2,
 'captures':1,'celebration':2,'celebrates':2,'affectionate':2,'affection':2,'fond':1,'gentle':1,'quietly':1,
 'believable':2,'convincing':2,'credible':2,'realistic':1,'natural':1,'lovable':2,'gratifying':2,
 'interesting':1,'worthwhile':2,'enough':0,'rewards':2,'reward':2,'achievement':2,'achieves':2,
 'succeed':2,'succeeded':2,'wonder':1,'wonders':2,'glad':2,'cheer':2,'cheerful':2,'cheering':2,'spark':1,
 'sparkling':2,'sparkle':2,'sparkles':2,'shines':2,'shine':2,'glow':1,'glowing':2,'gorgeously':3,
 'splendid':3,'splendidly':3,'thoroughly':1,'hugely':1,'immensely':1,'sumptuous':2,'exquisitely':3,
 'perfection':3,'honesty':2,'intimate':1,'intimacy':1,'unpretentious':2,'understated':1,'gentle':1,
 'important':1,'fantastic':3,'fabulous':3,'amazing':3,'amazingly':3,'incredible':2,'incredibly':1,
 'outstanding':3,'first-rate':3,'top-notch':3,'classic':2,'landmark':2,'milestone':2,'legendary':2,
 'bravura':2,'virtuoso':2,'tour':0,'force':0,'knockout':2,'dazzle':2,'dazzles':2,'charms':2,'delights':2,
 'satisfy':2,'satisfies':2,'satisfied':2,'enjoyment':2,'fulfilling':2,'lovingly':2,'loving':2,'tenderness':2,
 'enchant':2,'enchants':2,'seduces':2,'seductive':2,'alluring':2,'enticing':2,'tantalizing':2,'juicy':1,
 'tasty':2,'delicious':2,'savory':2,'feast':2,'spellbinding':3,'hypnotic':2,'haunted':1,'hauntingly':2,
 'eloquent':2,'eloquently':2,'lyrical':2,'poetic':2,'poetry':2,'graceful':2,'majestic':2,'epic':1,'grand':1,
 'admire':2,'admiration':2,'respect':1,'respectful':1,'proud':1,'worthy':2,'deserves':1,'earns':1,'earned':1,
 'realized':1,'fully':1,'ably':2,'capably':2,'nimble':2,'nimbly':2,'agile':2,'buoyant':2,'breezy':2,'zippy':2,
 'zesty':2,'punchy':2,'snappy':2,'slick':1,'tight':1,'tightly':1,'focused':1,'precise':1,'precision':1,
 'exciting':2,'rousing':2,'electric':2,'kinetic':1,'visceral':1,'raw':1,'gutsy':2,'fearless':2,'brave':2,
 'courageous':2,'heroic':1,'heartbreaking':1,'heart-breaking':1,'tearjerker':1,'cathartic':2,'devastating':1,
 'unsettling':1,'disturbing':0,'creepy':1,'chilling':1,'scary':1,'frightening':1,'spooky':1,'eerie':1,
 'passion':2,'passionate':2,'fervent':1,'ardent':1,'sympathetic':2,'sympathy':1,'empathy':2,'humane':2,
 'humanity':2,'human':1,'dignified':2,'noble':2,'tender-hearted':2,'kind':1,'kindness':2,'sweetness':2,
 'optimistic':2,'optimism':2,'uplift':2,'elevates':2,'elevated':1,'transcends':2,'transporting':2,
 'perceptive':2,'observant':2,'astute':2,'shrewd':2,'incisive':2,'penetrating':2,'piercing':1,'keen':2,
 'cinematic':1,'visually':1,'visual':0,'spectacle':1,'well':1,'nifty':2,'neat':1,'swell':2,'dandy':2,
 'likably':2,'relatable':2,'recognizable':1,'universal':1,'timeless':2,'enduring':2,'lasting':1,
 'surprise':1,'surprises':1,'twist':0,'twists':0,'imagination':1,'invention':1,'inventiveness':2,
 'laugh-out-loud':3,'uproarious':3,'uproariously':3,'side-splitting':3,'riotous':2,'riotously':2,
 'joyful':2,'joyfully':2,'exultant':2,'ecstatic':2,'euphoric':2,'giddy':1,'gleeful':2,'glee':2,
 'faithful':1,'sensitive':2,'sensitively':2,'nuance':2,'subtly':1,'subtlety':2,'restraint':1,'restrained':1,
 'intelligently':2,'cleverly':2,'smartly':2,'wittily':2,'sharply':1,'keenly':1,'shrewdly':2,'artfully':2,
 'elegantly':2,'gracefully':2,'beautifully':3,'handsomely':2,'handsome':2,'pretty':1,'lovely':2,'finely':2,
 'effortless':2,'effortlessly':2,'seamless':2,'seamlessly':2,'fluid':2,'fluidly':2,'smooth':1,'smoothly':1,
 'thrive':1,'thrives':1,'flourish':1,'flourishes':1,'blossom':1,'blossoms':1,'bloom':1,'blooms':1,
 'charmer':2,'charismatic':2,'charisma':2,'magnetic':2,'commanding':2,'riveted':2,'rivets':2,'enthralls':2,
 'spellbound':2,'mesmerize':2,'mesmerizes':2,'hooked':1,'hooks':1,'grabs':1,'grips':1,'absorbed':1,
 'entranced':2,'entrancing':2,'beguiling':2,'beguiles':2,'bewitching':2,'disarming':2,'disarmingly':2,
 'appreciate':1,'appreciated':1,'appreciation':1,'cherish':2,'cherished':2,'savor':2,'savored':2,
 'improves':1,'improvement':1,'improved':1,'enhances':1,'elevate':1,'enriches':2,'enriching':2,
 'nostalgic':1,'nostalgia':1,'fondly':2,'lovingly':2,'homage':1,'tribute':1,'honors':1,'honor':1,
 'liked':2,'like':0,'likes':1,'favorite':2,'favorites':2,'recommended':2,'recommendation':2,
 'moments':0,'pleasantly':2,'agreeable':2,'amiable':2,'genial':2,'affable':2,'congenial':2,'cozy':1,
 'satisfyingly':2,'comfortably':1,'comfortable':1,'relaxed':1,'laid-back':1,'easygoing':1,'mellow':1,
 'exceptional':3,'exceptionally':3,'extraordinarily':3,'superbly':3,'magnificently':3,'stunningly':3,
 'tremendous':3,'tremendously':2,'enormous':1,'enormously':1,'vast':1,'immense':1,'colossal':1,
 'succeeding':1,'victorious':2,'victory':2,'conquers':1,'prevails':1,'triumphs':2,'triumphantly':2,
 'skill':1,'skilled':2,'talent':1,'talented':2,'gifted':2,'gift':1,'expert':1,'expertly':2,'masterly':3,
 'craft':1,'crafted':1,'craftsmanship':2,'meticulous':2,'meticulously':2,'painstaking':1,'careful':1,
 'engagingly':2,'absorbingly':2,'compellingly':2,'grippingly':2,'movingly':2,'touchingly':2,'poignantly':2,
 'irresistibly':3,'infectiously':2,'exuberantly':2,'vividly':2,'richly':2,'lushly':2,'sumptuously':2,
 'refreshingly':2,'invigoratingly':2,'bracingly':2,'brightly':1,'bright':1,'sunny':1,'cheery':2,
 'nourishing':2,'nourishes':2,'sustaining':1,'sustains':1,'resonate':2,'resonating':2,'echoes':0,
}
NEG = {
 # strength 3
 'worst':3,'terrible':3,'awful':3,'horrible':3,'dreadful':3,'atrocious':3,'abysmal':3,'appalling':3,
 'unwatchable':3,'unbearable':3,'excruciating':3,'excruciatingly':3,'painful':2,'painfully':2,'torture':3,
 'garbage':3,'trash':3,'junk':3,'disaster':3,'disastrous':3,'catastrophe':3,'fiasco':3,'debacle':3,
 'stinks':3,'stinker':3,'stink':3,'sucks':3,'suck':3,'crap':3,'crappy':3,'drivel':3,'dreck':3,'turkey':3,
 'bomb':3,'dud':3,'flop':3,'mess':3,'messy':2,'waste':3,'wasted':2,'wastes':2,'worthless':3,'pointless':3,
 'laughable':3,'ludicrous':3,'ridiculous':3,'absurd':2,'idiotic':3,'moronic':3,'stupid':3,'dumb':3,'dumbest':3,
 'inept':3,'incompetent':3,'amateurish':3,'clumsy':2,'clumsily':2,'sloppy':2,'lazy':2,'lazily':2,'lousy':3,
 'pathetic':3,'miserable':3,'misery':2,'wretched':3,'vile':3,'offensive':3,'repugnant':3,'repellent':3,
 'disgusting':3,'revolting':3,'nauseating':3,'sickening':3,'loathsome':3,'hateful':3,'odious':3,'obnoxious':3,
 'insufferable':3,'intolerable':3,'unfunny':3,'witless':3,'brainless':3,'mindless':2,'soulless':3,'lifeless':3,
 'boring':3,'bore':3,'bored':3,'bores':3,'tedious':3,'tedium':3,'dull':3,'dullest':3,'dreary':3,'drab':2,
 'insipid':3,'vapid':3,'inane':3,'banal':3,'trite':3,'hackneyed':3,'stale':2,'tired':2,'tiresome':3,
 'lame':3,'feeble':3,'flimsy':2,'limp':2,'listless':3,'leaden':3,'plodding':3,'sluggish':2,'turgid':3,
 'bland':3,'tasteless':2,'flavorless':2,'colorless':2,'toothless':2,'spineless':2,'gutless':2,
 'failure':3,'fails':3,'fail':3,'failed':3,'disappointing':3,'disappointment':3,'disappoints':3,'disappointed':3,
 'botched':3,'bungled':3,'botch':3,'ruins':2,'ruined':2,'ruin':2,'squanders':3,'squandered':3,'squander':3,
 'forgettable':3,'unmemorable':3,'mediocre':3,'mediocrity':3,'lackluster':3,'uninspired':3,'uninspiring':3,
 'unimaginative':3,'unoriginal':3,'derivative':3,'formulaic':3,'formula':2,'predictable':3,'cliche':3,
 'cliches':3,'clichéd':3,'cliched':3,'stereotypical':2,'stereotypes':2,'generic':2,'cookie-cutter':3,
 'contrived':3,'forced':2,'strained':2,'labored':2,'heavy-handed':3,'ham-fisted':3,'ham-handed':3,
 'overwrought':2,'overblown':2,'bloated':3,'overlong':3,'interminable':3,'endless':2,'drags':3,'dragged':2,
 'drag':2,'dragging':2,'slog':3,'chore':3,'ordeal':3,'snooze':3,'snoozer':3,'yawn':3,'sleep':1,'asleep':2,
 'shallow':3,'hollow':3,'empty':2,'vacuous':3,'superficial':3,'thin':2,'slight':2,'flat':2,'flatly':2,
 'pretentious':3,'pompous':3,'self-indulgent':3,'indulgent':2,'smug':3,'preachy':3,'preaching':2,
 'manipulative':3,'cynical':2,'cynically':2,'exploitative':3,'sleazy':2,'tacky':2,'cheap':2,'cheesy':2,
 'corny':2,'hokey':2,'sappy':2,'saccharine':3,'schmaltzy':3,'maudlin':3,'mawkish':3,'syrupy':2,
 'sentimental':1,'melodramatic':2,'melodrama':1,'soap':1,'tearjerking':1,'treacly':3,'cloying':3,
 'annoying':3,'irritating':3,'grating':3,'aggravating':3,'exasperating':3,'infuriating':3,'maddening':3,
 'frustrating':3,'tiring':2,'exhausting':2,'numbing':3,'deadening':3,'stultifying':3,'soporific':3,
 'bad':3,'badly':2,'poor':3,'poorly':3,'weak':3,'weakest':3,'worse':2,'lesser':1,'inferior':2,
 'wrong':2,'misguided':3,'misfire':3,'misfires':3,'miscalculation':3,'mistake':2,'mistakes':1,'error':1,
 'problem':2,'problems':2,'problematic':2,'flawed':2,'flaws':2,'flaw':2,'defects':2,'shortcomings':2,
 'lacks':2,'lack':2,'lacking':2,'missing':2,'misses':2,'miss':2,'missed':2,'loses':2,'lost':2,'lose':1,
 'incoherent':3,'confusing':2,'confused':2,'muddled':3,'muddle':3,'jumbled':2,'disjointed':2,'incomprehensible':3,
 'nonsense':3,'nonsensical':3,'silly':2,'sillier':2,'silliness':2,'goofy':1,'juvenile':3,'sophomoric':3,
 'childish':3,'infantile':3,'puerile':3,'adolescent':1,'immature':2,'crude':2,'vulgar':2,'tawdry':3,
 'ugly':3,'hideous':3,'grotesque':2,'unpleasant':3,'unsavory':2,'distasteful':3,'tasteless':2,
 'dismal':3,'grim':1,'bleak':1,'depressing':2,'dour':2,'gloomy':1,'somber':0,'dark':0,'morose':2,
 'hate':3,'hated':3,'hates':2,'despise':3,'loathe':3,'detest':3,'abhor':3,'dislike':2,'disliked':2,
 'irrelevant':2,'useless':3,'meaningless':3,'purposeless':3,'aimless':3,'rudderless':3,'directionless':3,
 'unconvincing':3,'implausible':2,'unbelievable':1,'far-fetched':2,'preposterous':3,'inconsistent':2,
 'unfocused':3,'rambling':2,'meandering':2,'meanders':2,'wanders':2,'wandering':1,'scattered':2,'scattershot':3,
 'awkward':2,'awkwardly':2,'stilted':3,'wooden':3,'stiff':2,'mechanical':2,'rote':2,'perfunctory':3,
 'phony':3,'fake':2,'false':2,'bogus':3,'hokum':3,'hooey':3,'bull':2,'baloney':3,'claptrap':3,'tripe':3,
 'dumbed-down':3,'dumbed':2,'simplistic':3,'simple-minded':3,'one-dimensional':3,'two-dimensional':2,'cardboard':3,
 'cartoonish':2,'caricature':2,'caricatures':2,'cartoon':0,'unlikable':3,'unlikeable':3,'unsympathetic':2,
 'repetitive':3,'repetitious':3,'redundant':2,'monotonous':3,'monotone':2,'droning':2,'drone':2,'drones':2,
 'overdone':2,'overacted':2,'overacting':3,'overwrought':2,'overbearing':2,'oppressive':2,'ponderous':3,
 'lumbering':3,'lumbers':2,'clunky':3,'clunker':3,'creaky':2,'creaks':2,'rusty':2,'dated':2,'outdated':2,
 'exercise':1,'excuse':2,'gimmick':2,'gimmicky':3,'gimmicks':2,'schtick':1,'shtick':1,'ploy':2,'cheat':2,
 'cheats':2,'cheated':2,'robbed':2,'rip-off':3,'ripoff':3,'rehash':3,'rehashed':3,'recycled':3,'retread':3,
 'knockoff':3,'knock-off':3,'imitation':2,'copy':1,'carbon':1,'xerox':2,'warmed-over':3,'leftovers':3,
 'disjointed':2,'incoherence':3,'coherence':0,'rickety':2,'shaky':2,'uneven':2,'unsteady':1,'wobbly':2,
 'listlessly':3,'lethargic':3,'lethargy':3,'inert':3,'inertia':2,'static':2,'stagnant':3,'stagnates':3,
 'bores':3,'ennui':3,'sleepy':2,'sleep-inducing':3,'somnolent':3,'anesthetic':3,'lobotomized':3,
 'loud':1,'noisy':2,'noise':2,'shrill':3,'strident':2,'screechy':3,'grates':3,'grating':3,'jarring':2,
 'obvious':2,'obviously':1,'blatant':2,'heavy':1,'heavily':1,'overstuffed':2,'cluttered':3,'clutter':2,
 'unnecessary':2,'gratuitous':2,'excessive':2,'excess':1,'overkill':2,'overload':2,'too':1,'much':0,
 'ugh':3,'yuck':3,'blah':3,'meh':3,'bleh':3,'zzz':3,'eh':1,'nah':2,'nope':2,'whatever':1,
 'offend':2,'offends':2,'offended':2,'insult':3,'insulting':3,'insults':3,'condescending':3,'patronizing':3,
 'mean-spirited':3,'nasty':2,'cruel':2,'callous':2,'cold':2,'chilly':2,'frigid':2,'icy':2,'clinical':1,
 'detached':2,'distant':2,'remote':2,'alienating':3,'off-putting':3,'unappealing':3,'uninviting':3,
 'bleakly':1,'drably':2,'dully':3,'blandly':3,'tediously':3,'tiresomely':3,'interminably':3,'endlessly':2,
 'unfortunately':2,'sadly':2,'regrettably':2,'alas':2,'pity':2,'shame':2,'shameful':3,'shamefully':3,
 'embarrassing':3,'embarrassment':3,'embarrassingly':3,'cringe':3,'cringe-inducing':3,'cringeworthy':3,
 'nothing':1,'none':1,'never':1,'nowhere':2,'nobody':1,'zero':1,'little':1,'barely':1,'hardly':1,
 'questionable':2,'dubious':2,'suspect':1,'unclear':1,'ambiguous':0,'murky':2,'muddy':2,'fuzzy':1,
 'disappointingly':3,'dismally':3,'poorly':3,'weakly':2,'feebly':3,'lamely':3,'limply':2,'flatly':2,
 'contrivance':3,'contrivances':3,'convolution':2,'convoluted':3,'byzantine':1,'labyrinthine':1,
 'fatigue':2,'fatigued':2,'tiredness':2,'weary':2,'wearying':3,'wearisome':3,'wears':2,'wear':1,
 'overrated':3,'hyped':2,'overhyped':3,'hype':2,'bland':3,'anemic':3,'bloodless':3,'listless':3,
 'cheapens':3,'cheapened':3,'degrades':3,'debases':3,'demeans':3,'demeaning':3,'degrading':3,
 'exploits':2,'exploitation':2,'pandering':3,'panders':3,'pander':3,'lowest':2,'denominator':2,
 'disjointedly':2,'haphazard':3,'haphazardly':3,'slapdash':3,'half-baked':3,'half-hearted':3,'halfhearted':3,
 'misstep':3,'missteps':3,'stumbles':2,'stumble':2,'falters':2,'falter':2,'flounders':3,'flounder':3,
 'collapses':3,'collapse':3,'crumbles':3,'crumble':2,'unravels':2,'unravel':2,'implodes':3,'sinks':2,'sink':2,
 'mishmash':3,'hodgepodge':3,'jumble':2,'grab-bag':2,'patchwork':2,'pastiche':1,'mush':3,'mushy':2,
 'hollowness':3,'emptiness':3,'shallowness':3,'banality':3,'triteness':3,'staleness':3,'dullness':3,
 'disbelief':1,'unfulfilled':3,'unfulfilling':3,'unsatisfying':3,'unsatisfied':2,'unrewarding':3,
 'tepid':3,'lukewarm':3,'cool':0,'mild':1,'mildly':1,'faint':1,'faintly':1,'pale':2,'wan':2,'washed-out':2,
 'squandering':3,'squanders':3,'wasting':3,'misuse':2,'misused':2,'misuses':2,'underused':2,'underutilized':2,
 'unsure':1,'uncertain':1,'hesitant':1,'timid':2,'tentative':2,'cautious':1,'safe':1,'bland':3,
 'exhausted':2,'exhaustion':2,'spent':1,'depleted':2,'drained':2,'draining':2,'sapped':2,'saps':2,
}
NEGATORS = {'not':1.0,"n't":1.0,'never':1.0,'no':0.8,'nothing':0.8,'without':0.8,'hardly':0.8,'barely':0.8,
            'neither':0.8,'nor':0.8,'lacks':0.6,'lack':0.6,'fails':0.6,'cannot':1.0,"can't":1.0,'isnt':1.0,
            'none':0.8,'nobody':0.8,'rarely':0.6,'seldom':0.6,'scarcely':0.8,'devoid':0.8,'free':0.0}
INTENSIFIERS = {'very':0.5,'extremely':0.8,'incredibly':0.6,'remarkably':0.5,'truly':0.5,'really':0.4,
                'so':0.3,'absolutely':0.7,'utterly':0.7,'completely':0.6,'totally':0.6,'thoroughly':0.5,
                'deeply':0.4,'highly':0.5,'most':0.4,'especially':0.3,'particularly':0.3,'exceptionally':0.7,
                'amazingly':0.6,'surprisingly':0.3,'wonderfully':0.5,'terribly':0.4,'awfully':0.4,
                'quite':0.3,'such':0.3,'damn':0.5,'pretty':0.2,'genuinely':0.4,'profoundly':0.6,'richly':0.4,
                'enormously':0.5,'immensely':0.5,'hugely':0.5,'wildly':0.5,'entirely':0.5,'too':0.3,'overly':0.3}
CONTRAST = {'but':1.0,'however':1.0,'yet':0.8,'although':0.6,'though':0.5,'despite':0.6,'unfortunately':0.7,
            'except':0.7,'nevertheless':0.8,'nonetheless':0.8,'still':0.4,'ultimately':0.6,'instead':0.5,
            'while':0.3,'whereas':0.6,'otherwise':0.5,'unless':0.5,'only':0.3,'until':0.3}
HEDGES = {'somewhat':0.5,'slightly':0.6,'a bit':0.5,'bit':0.4,'kind':0.3,'sort':0.3,'rather':0.2,'fairly':0.2,
          'mildly':0.6,'occasionally':0.4,'sometimes':0.4,'almost':0.4,'nearly':0.3,'merely':0.5,'just':0.2,
          'only':0.3,'modest':0.4,'modestly':0.4,'moderately':0.5,'passable':0.4,'passably':0.4,'ok':0.3,
          'okay':0.3,'decent':0.2,'adequate':0.4,'serviceable':0.4,'competent':0.3,'watchable':0.3,'harmless':0.5,
          'inoffensive':0.5,'tolerable':0.5,'acceptable':0.4,'average':0.5,'ordinary':0.5,'routine':0.5,
          'middling':0.6,'so-so':0.7,'mixed':0.6,'uneven':0.5}

def build_embedding():
    E = {}
    def row(w):
        if w not in E: E[w] = np.zeros(D)
        return E[w]
    for w,s in POS.items(): row(w)[0] = max(row(w)[0], s)
    for w,s in NEG.items(): row(w)[1] = max(row(w)[1], s)
    for w,s in NEGATORS.items(): row(w)[2] = s
    for w,s in INTENSIFIERS.items(): row(w)[3] = s
    for w,s in CONTRAST.items(): row(w)[4] = s
    for w,s in HEDGES.items(): row(w)[5] = s
    # hedged-positive words carry a small positive charge (mildly good) rather than none
    for w in ['decent','competent','watchable','passable','ok','okay','adequate','serviceable','harmless','inoffensive']:
        row(w)[0] = max(row(w)[0], 0.6)
    for w in ['average','ordinary','routine','middling','so-so','mixed','tolerable','acceptable']:
        row(w)[1] = max(row(w)[1], 0.9)
    return E
E = build_embedding()
UNK = np.zeros(D)

# ---------------------------------------------------------------------------
# 2. RECURRENT CELL (hand-written weights)
#    h = [pos_acc, neg_acc, neg_gate, int_gate]
# ---------------------------------------------------------------------------
# gate recurrences: neg_gate_t = clip(e_neg + W_gg[0]*neg_gate_{t-1}, 0, 1)
W_gate_rec = np.array([0.55, 0.45])          # decay of negation gate, intensifier gate per token
W_gate_in  = np.array([1.0, 1.0])            # input weight from embedding axes 2,3
# contrast gate: when a contrast word appears, previous accumulators are scaled down
W_contrast_decay = 0.35
# hedge: scales the current token's sentiment
W_hedge = 1.0
# negation: flipped polarity carries this fraction of original strength ("not great" ~ mildly negative)
W_flip_pos2neg = 0.65   # "not good"  -> negative with 0.65 * strength
W_flip_neg2pos = 0.45   # "not bad"   -> positive with 0.45 * strength
# intensifier: scale = 1 + W_int * int_gate
W_int = 1.0
# accumulation: running sum with a mild forgetting factor so late sentences dominate a touch
W_acc_rec = 0.97

def relu(x): return np.maximum(x, 0)
def clip01(x): return x - relu(x - 1.0)      # clip to [0,1] built from ReLU

def cell(h, e):
    pos_acc, neg_acc, ng, ig = h
    e_pos, e_neg, e_negator, e_int, e_con, e_hedge = e
    # 1. contrast: discount history
    scale_hist = 1.0 - W_contrast_decay*0 if e_con == 0 else (1.0 - e_con*(1-W_contrast_decay))
    pos_acc *= scale_hist * W_acc_rec
    neg_acc *= scale_hist * W_acc_rec
    # 2. gates (ReLU-clipped recurrences)
    ng_new = clip01(W_gate_in[0]*e_negator + W_gate_rec[0]*ng)
    ig_new = clip01(W_gate_in[1]*e_int     + W_gate_rec[1]*ig)
    # 3. polarity routing under negation (the gate that was active BEFORE this token)
    g = ng   # negation applies to following words, not itself
    pos_eff = (1-g)*e_pos + g*W_flip_neg2pos*e_neg
    neg_eff = (1-g)*e_neg + g*W_flip_pos2neg*e_pos
    # 4. intensity & hedging
    amp = (1.0 + W_int*ig) * (1.0 - W_hedge*0.5*e_hedge)
    pos_acc += amp*pos_eff
    neg_acc += amp*neg_eff
    return np.array([pos_acc, neg_acc, ng_new, ig_new])

# ---------------------------------------------------------------------------
# 3. READOUT  Linear(4->6) + ReLU + Linear(6->1)
# ---------------------------------------------------------------------------
W1 = np.array([
 [ 1.0, -1.0, 0.0, 0.0],   # net sentiment
 [-1.0,  1.0, 0.0, 0.0],   # net negativity
 [ 1.0,  0.0, 0.0, 0.0],   # any positivity
 [ 0.0,  1.0, 0.0, 0.0],   # any negativity
 [ 1.0,  1.0, 0.0, 0.0],   # total emotional mass (both present -> ambivalence)
 [ 0.0,  0.0, 1.0, 0.0],   # dangling negation at end ("... not .") hints negative
])
b1 = np.zeros(6)
W2 = np.array([ 1.2, -1.2, 0.15, -0.35, -0.05, -0.4 ])
b2 = -0.15   # reviewers write "is fine" rarely; neutral-ish sentences lean slightly negative in SST

def forward(tokens):
    h = np.zeros(4)
    for t in tokens:
        h = cell(h, E.get(t, UNK))
    z = relu(W1 @ h + b1)
    logit = W2 @ z + b2
    return 1/(1+np.exp(-logit))

def tokenize(s):
    s = s.lower().replace("n't", " n't").replace("' t", " n't")
    return re.findall(r"[a-z'\-]+|[.,!?]", s)

def predict(sentence):
    return int(forward(tokenize(sentence)) >= 0.5)

def count_params():
    n_emb = sum(int((v != 0).sum()) for v in E.values())
    n_cell = W_gate_rec.size + W_gate_in.size + 5   # scalars
    n_read = int((W1!=0).sum()) + int((W2!=0).sum()) + 1
    return n_emb, n_cell, n_read

if __name__ == '__main__':
    import sys
    data = sys.argv[1]
    X, y = [], []
    for line in open(data):
        line = line.rstrip('\n')
        if not line: continue
        s, lab = line.rsplit('\t', 1)
        X.append(s); y.append(int(lab))
    preds = [predict(s) for s in X]
    acc = np.mean(np.array(preds) == np.array(y))
    ne, nc, nr = count_params()
    print(f"{data}: n={len(y)} accuracy={acc*100:.1f}%  nonzero params: embedding={ne} cell={nc} readout={nr} total={ne+nc+nr}")
    tp = sum(p==1 and t==1 for p,t in zip(preds,y)); fp = sum(p==1 and t==0 for p,t in zip(preds,y))
    fn = sum(p==0 and t==1 for p,t in zip(preds,y)); tn = sum(p==0 and t==0 for p,t in zip(preds,y))
    print(f"  confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
