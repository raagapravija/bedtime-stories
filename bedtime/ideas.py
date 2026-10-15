"""The agent's sources for story ideas: a premise bank, a space fact bank and
a few comic scenes.

Why: gpt-3.5 is a decent developer of a good idea and a poor inventor of one.
Asked for "an adventure for a 7-year-old" it wrote a puzzle that fell into other
puzzles; asked for space, "Hana goes to the Moon and meets an alien". Agents'
Room (Huot et al., ICLR 2025) showed that story quality comes from good planning
material that specialists build on - so the planning specialists here start
from concrete, human-written seeds instead of a blank page.

Every seed is original and has the same parts: a hero who WANTS something
specific, an obstacle with its own funny logic, and a twist. The agent adapts
seeds to the child (name, today's moment) and may combine them; it never has
to use one as written.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Seed:
    key: str
    title: str
    hook: str        # who, what they want, what is in the way (and why it is funny)
    twist: str       # the turn near the end
    ages: tuple = (5, 10)
    hero: str = ""   # what the hero IS, when it isn't a child ("a little flame inside the Sun")
    learn: str = ""  # the true thing the listener finds out, in kid words


def _s(key, title, hook, twist, ages=(5, 10)):
    return Seed(key, title, hook, twist, ages)


def _w(key, title, hero, hook, learn, twist, ages=(5, 10)):
    """A wonder seed: the hero LIVES the science, and the story is about finding it out."""
    return Seed(key, title, hook, twist, ages, hero, learn)


SEEDS: dict[str, list[Seed]] = {
    "fairy_tale": [
        _s("duck_knight", "The Knight Who Was Afraid of Ducks",
           "Sir Barnaby must carry the queen's birthday crown across the royal duck pond, but he is "
           "terribly afraid of ducks - and the ducks think his shiny helmet is a giant bread roll.",
           "The ducks were never chasing him: they wanted to march beside him like a parade."),
        _s("pancake_tower", "The Tower Made of Pancakes",
           "By mistake, the royal cook built the princess's tower out of pancakes. Every knight who "
           "climbs up to fetch her for breakfast gets hungry and eats a step.",
           "The princess comes down by herself, eating her way, with a jug of syrup."),
        _s("frog_no", "The Frog Who Didn't Want to Be a Prince",
           "Every princess in the land wants to kiss Frederick the frog and turn him into a prince. "
           "But Frederick loves being a frog, and he must find a polite way to say no.",
           "He teaches the princess to hop, and she likes it better than dancing."),
        _s("giant_sock", "The Giant's Lost Sock",
           "The giant's sock blows off his washing line and lands on the village as a huge striped tent. "
           "The villagers love their new tent - but the giant's foot is getting very cold.",
           "The whole village knits him a new sock, and the old one stays a tent for parties."),
        _s("mishearing_well", "The Wishing Well That Mishears",
           "The old wishing well grants every wish, but it is a little deaf: 'a pony' became 'a pastry'. "
           "Tom needs rain for his grandma's thirsty garden.",
           "He wishes for 'a train' on purpose - and a little cloud-train puffs over the garden, raining."),
        _s("queen_hiccups", "The Queen's Hiccups",
           "The queen has the hiccups on the day of her big speech. The whole kingdom tries to cure them: "
           "a fright, a spoon of sugar, standing on one leg.",
           "The smallest page tells her a terrible joke, and laughing cures them."),
        _s("gingerbread_map", "The Gingerbread Map",
           "The baker's girl bakes a treasure map into the king's gingerbread by mistake. They must find "
           "the treasure before the king eats the map - and he is very hungry.",
           "The treasure is the baker's lost recipe book, and the king gets a new gingerbread."),
        _s("slipper_mouse", "The Missing Glass Slipper",
           "The youngest of twelve princesses has lost one slipper before the royal dance, and she will "
           "have to hop all evening unless she finds it.",
           "A mouse borrowed it as a bed for her babies, and the princess dances happily in her socks."),
    ],
    "magical": [
        _s("indoor_umbrella", "The Umbrella That Rains Indoors",
           "Zara's new umbrella keeps her dry when it is open, but it rains whenever it is closed - and "
           "today she must carry it, closed, all the way through the school show.",
           "Backstage it waters the class's droopy sunflowers just in time for the last song."),
        _s("runaway_shadow", "The Shadow That Ran Off to Play",
           "Leo's shadow gets bored of copying him and runs off to play tag with the other shadows. "
           "Leo must catch it before the school photo.",
           "Leo stops chasing and starts playing tag too - and his shadow comes running back."),
        _s("tiny_wishes", "The Teapot of Tiny Wishes",
           "Granny's teapot grants wishes, but only tiny ones: a crumb bigger, a sock warmer, a cloud "
           "rounder. Granny's cat is stuck on the shed roof.",
           "Lots of tiny wishes together - a step taller, a ladder longer - get the cat down."),
        _s("paint_alive", "The Paint Box That Brings Things to Life",
           "Anything Ana paints comes alive for one hour. She paints a cat to play with, but forgot to "
           "paint its tail, and the cat is very upset about it.",
           "She paints it the longest, stripiest tail in the world, and the cat is very proud."),
        _s("booming_book", "The Book That Reads Itself Aloud",
           "Mia's library book reads itself aloud in a big booming voice - in the quietest library in town. "
           "She must hush it before the librarian notices.",
           "The librarian loves it, and the book gets its own story time on Fridays."),
        _s("snow_globe", "Grandpa's Weather Globe",
           "Shaking Grandpa's snow globe changes the weather outside. On picnic day, little brother Sam "
           "shakes it far too hard.",
           "They have the first ever picnic in the snow, with hot soup, and Sam says sorry to the sun."),
        _s("sneeze_rabbits", "The Hat Full of Rabbits",
           "The magician's old hat makes a rabbit pop out every time someone sneezes - and it is "
           "hay fever season at the village fair.",
           "The rabbits pull the magician's cart home, and he finally gets a holiday."),
        _s("backwards_minute", "The Backwards Minute",
           "Mum's old clock can make one minute run backwards. Kai has just spilled jam on Grandma's "
           "birthday tablecloth.",
           "Running the minute back also un-eats his toast, so he gets breakfast twice."),
    ],
    "calming": [
        _s("night_bakers", "The Night Bakers",
           "Every night the mouse family bakes the town's bread. Tonight the littlest mouse must wake the "
           "oven cat - without waking the baby.",
           "The cat and the baby both sleep on, and the little mouse lights the oven all by herself."),
        _s("snail_goodnights", "The Snail Who Delivered Goodnights",
           "Sheldon the snail delivers goodnight notes to everyone in the garden, but he is very slow. "
           "He sets off at breakfast so the last note arrives at bedtime.",
           "The last note is for him, from everyone he visited."),
        _s("sleepy_train", "The Sleepy Train",
           "The last train of the night carries sleepy animals home. One sheep keeps getting off at the "
           "wrong stop because she is counting people instead of stations.",
           "The driver lets her count the stations out loud, and she is the last one home, smiling."),
        _s("quilt_map", "Grandma's Quilt Map",
           "Every patch of Grandma's quilt is a place she once visited. Tonight, before sleep, Noor picks "
           "three patches and Grandma tells what happened there.",
           "The newest patch is Noor's own pyjama fabric: their trip to the seaside last summer."),
        _s("boot_story", "The Boots Remember",
           "After a rainy day of puddle jumping, Ben's boots by the door talk quietly about the day: the "
           "biggest puddle, the worm, the splash on Dad.",
           "They agree tomorrow's puddles will be even better, and they fall asleep too."),
        _s("lighthouse_cat", "The Lighthouse Cat",
           "The lighthouse keeper has a cold, so her cat Biscuit must help her keep the lamp going and "
           "count the fishing boats safely home.",
           "Every boat comes home, and the fishermen leave Biscuit a fish on the doorstep."),
        _s("counting_boats", "Counting Boats",
           "Ellie can't fall asleep, so Dad says to count the little boats in the harbour. But one boat "
           "keeps waving back.",
           "It is her uncle's boat, coming home late, flashing goodnight with his torch."),
        _s("bat_sunshine", "The Bat Who Wanted Sunshine",
           "Pip the bat sleeps all day and has never seen the sun. Her friend the hedgehog promises to "
           "wake her up for one sunrise.",
           "They watch the sunrise together - then both yawn and go to bed at breakfast time."),
    ],
    "good_manners": [
        _s("gimme_dragon", "The Dragon Who Said Gimme",
           "Drago toasts the best marshmallows in the valley, but nobody shares theirs with him because he "
           "always says 'Gimme!'. He tries every other word he knows.",
           "The word that works is 'please' - and then everyone wants his toasting too."),
        _s("line_penguin", "The Penguin Who Cut the Line",
           "Pablo the penguin always cuts in line at the ice-cream iceberg - until the day the line is "
           "so long it goes all the way round the iceberg and he can't find the front.",
           "He ends up at the back behind his grandma, who saves him the last cone."),
        _s("thank_you_avalanche", "The Thank-You Note Avalanche",
           "Kiko decides to write a thank-you note to everyone and everything: the bus driver, the rain, "
           "the toaster. Then things start writing back.",
           "The toaster's note says 'You're welcome', burnt into a slice of toast."),
        _s("knocking_elephant", "The Elephant Who Forgot to Knock",
           "Ella the elephant always forgets to knock and walks straight in. When she finally remembers, "
           "her knock shakes the whole house.",
           "Her friends give her a tiny bell to ring instead, and it becomes her favourite thing."),
        _s("sorry_squirrel", "Sorry, Said the Squirrel",
           "Sid the squirrel took everyone's acorns by mistake. Saying sorry to each animal costs him "
           "something funny: a dance, a song, a ride on his tail.",
           "The last animal, Owl, only wants a hug - and gives the acorns back."),
        _s("please_robot", "The Robot Who Said Please Too Much",
           "Bolt the robot says please four hundred times a day: please to the door, please to the stairs, "
           "please to his own feet. He must learn when it matters.",
           "He says one perfect please to the baker and gets a whole cake."),
        _s("tiger_tea", "Table Manners for a Tiger",
           "Tilly invites a tiger to tea, and the tiger tries very hard to be polite: napkin, small bites, "
           "no roaring. But the cake is very, very good.",
           "Tilly teaches him one polite roar for 'this is delicious'."),
        _s("swing_turn", "The Hedgehog's Turn",
           "Hettie the hedgehog waits for her turn on the swing, but everyone counts to ten too fast and "
           "starts again.",
           "She teaches everyone to count with clapping, and the turns finally come round."),
    ],
    "healthy_habits": [
        _s("toothbrush_strike", "The Toothbrush Who Went on Strike",
           "Max's toothbrush refuses to work until Max brushes properly: two whole minutes, every tooth. "
           "They argue, then make a deal.",
           "They make up a two-minute brushing song, and the toothbrush sings the chorus."),
        _s("carrot_rocket", "Carrot Rocket Fuel",
           "Juno won't eat vegetables unless they are rocket fuel. So Dad builds a rocket menu: launch "
           "peas, booster carrots, moon-rock potatoes.",
           "Juno's rocket 'flies' all the way to pudding, and she asks for more booster carrots."),
        _s("sleepy_superhero", "The Sleepy Superhero",
           "Captain Zoom stays up too late every night, and today she is too sleepy to rescue a kitten "
           "from a low branch.",
           "After one good night's sleep, she rescues the kitten before breakfast - sleep is her super power."),
        _s("soap_army", "The Bubble Army",
           "Leo's hands want to stay muddy forever. The soap bubbles line up like a tiny army to win them back.",
           "The bubbles win with tickling, and Leo's hands surrender, giggling."),
        _s("sneeze_patrol", "Sneeze Patrol",
           "Sneezes in Mia's house are tiny escaping creatures. Mia joins the Sneeze Patrol to catch them "
           "in elbows before they get away.",
           "She catches the biggest sneeze of all: Grandpa's."),
        _s("bedtime_race", "Grandpa's Bedtime Race",
           "Ruby challenges Grandpa to a bedtime race: teeth, pyjamas, story. Grandpa keeps losing in "
           "very silly ways.",
           "Ruby notices he loses on purpose, and lets him win the last one."),
        _s("water_camel", "The Camel Who Forgot to Drink",
           "Carlos the camel thinks camels never need water, so he skips drinks on the long walk to the fair. "
           "Soon he is very wobbly.",
           "One long drink at the well, and he wins the camel race."),
        _s("mud_monster_bath", "Bath Time for a Mud Monster",
           "A small mud monster thinks mud IS clean. His friend has to convince him to try one bath.",
           "He loves the bubbles so much he becomes a bubble monster."),
    ],
    # Real animals with true, surprising habits (the same "learn something" idea
    # as space): the habit is what the story is about.
    "animal_friends": [
        _w("otter_hands", "The Otter Who Drifted Off", "a young sea otter",
           "Every night the little otter falls asleep floating on her back - and wakes up somewhere new: next "
           "to a grumpy seal, under a fishing boat, by the lighthouse. Her family has to come and find her "
           "every morning.",
           "Sea otters sleep floating on the water. They hold paws, or wrap themselves in seaweed, so they "
           "don't drift apart.",
           "Grandma otter shows her how to hold paws, and the whole family sleeps in a long floating line."),
        _w("octopus_hearts", "Three Hearts for Olive", "a young octopus",
           "Doctor Crab listens to the octopus's chest and hears ba-dum, ba-dum, ba-dum - three hearts! He is "
           "sure something is wrong and calls every doctor in the reef, and each one has a sillier idea.",
           "An octopus has three hearts.",
           "Grandma octopus laughs: every octopus has three hearts, and the little octopus is perfectly well."),
        _w("bee_dance", "The Bee Who Couldn't Dance", "a young honeybee",
           "The bee finds the best flower field ever and races home to tell the hive. But bees don't tell with "
           "words - they tell with a dance - and her wobbly dance sends everyone to a puddle, a dustbin and "
           "Grandpa's hat.",
           "Honeybees tell each other where the flowers are by dancing: the waggle dance points the way, and "
           "how long they waggle says how far.",
           "She practises with an old bee, dances it right, and the whole hive flies straight to the field."),
        _w("penguin_pebble", "The Perfect Pebble", "a young gentoo penguin",
           "The penguin wants to give his best friend the perfect pebble, because that is how penguins say "
           "'you matter to me'. He tries a rock that is too big, a pebble that is too small, and a bottle top "
           "that is not a pebble at all.",
           "Gentoo penguins give each other pebbles. They use them to build their nests, and a good pebble is "
           "a very special present.",
           "His friend has been looking for a perfect pebble for him too, and they swap at the same moment."),
        _w("dolphin_half_sleep", "The Dolphin Who Slept With One Eye Open", "a young dolphin",
           "At the reef sleepover, everyone tries to stay awake the longest. The dolphin is sure she will win, "
           "but then she feels half of her get very, very sleepy - only half.",
           "Dolphins sleep with half of their brain at a time, so one eye stays open and they can still swim "
           "up to breathe.",
           "She wins and falls asleep at the same time: one eye winks goodnight while the other keeps watch."),
        _w("green_sloth", "The Sloth Who Turned Green", "a young sloth",
           "The sloth moves so slowly that tiny green plants have started growing in his fur. Birds land on "
           "him, a frog asks to rent his ear, and his friends think he has turned into a bush.",
           "Sloths move so slowly that tiny green plants (algae) grow in their fur - which helps them hide in "
           "the trees.",
           "At the jungle's hide-and-seek game, nobody finds him for the whole afternoon."),
        _w("owl_head_turn", "I Spy With My Owl Eye", "a young owl",
           "The owl plays I Spy with her friends, but owls can't move their eyes, so she has to turn her whole "
           "head - and she turns it so far she spies things behind her that nobody else can see.",
           "Owls can't move their eyes, so they turn their whole head instead - almost all the way round.",
           "She wins I Spy by spying the moon rising behind everyone."),
        _w("elephant_feet", "Listening With Your Feet", "a young elephant",
           "The elephant keeps stamping, and everyone tells her to be quiet. But she isn't stamping - she is "
           "listening, because she can feel her grandma's rumble far across the grass.",
           "Elephants can 'hear' with their feet: they feel deep rumbles from other elephants through the "
           "ground, from far away.",
           "She feels the rumble and leads her lost friends straight to grandma's watering hole."),
        _s("tortoise_parcel", "The Slowest Delivery",
           "Tortoise is the forest's postman, and he must deliver a big birthday parcel before the party "
           "starts. Everyone keeps stopping him to chat.",
           "The parcel is for him: the party is his own surprise birthday."),
        _s("snoring_bear", "The Bear Who Snored the Forest Awake",
           "Bruno the bear snores so loudly the whole forest can't sleep. The animals try to help him "
           "snore quietly.",
           "A tiny mouse sings him a lullaby, and his snore turns into a soft hum."),
    ],
    "adventure": [
        _s("backwards_parrot", "The Parrot Who Speaks Backwards",
           "The town bell is missing on the morning of the big wedding. The only clue is the bell-ringer's "
           "parrot, who repeats everything backwards.",
           "Said forwards, the parrot's words lead to the bakery, where the bell is being used as a cake mould."),
        _s("cereal_map", "The Map on the Cereal Box",
           "The back of Maya's cereal box has a treasure map of her own street. On Saturday, she and Dad "
           "follow it: past the post box, under the big hedge, to the old oak.",
           "The treasure is a tin Dad buried when he was seven, with his old toy car inside."),
        _s("windy_hat", "Grandpa's Hat and the Windy Hill",
           "The wind takes Grandpa's favourite hat and keeps carrying it a little further: over the stream, "
           "across the sheep field, up the windy hill.",
           "At the top, a goat is wearing it. Grandpa lets him keep it and buys a new one."),
        _s("teddy_island", "The Toy Boat to Teddy Island",
           "Little Ben's teddy is stuck on the tiny island in the park pond. From the bank, big sister Ana and "
           "Mum send toy boats to fetch him - a paper boat, a lunchbox boat, a boot boat - but ducks keep "
           "climbing aboard.",
           "The ducks push the lunchbox boat to the island, and it comes back with Teddy sitting in it like a "
           "captain."),
        _s("snowman_nose", "The Case of the Missing Carrot Nose",
           "The snowman's carrot nose has vanished overnight. Small footprints lead around the garden, "
           "behind the shed and under the gate.",
           "A rabbit took it - and left the snowman a pine-cone nose instead, to say sorry."),
        _s("pumpkin_roll", "The Great Pumpkin Roll",
           "The prize pumpkin rolls off the cart at the top of the village hill, and everyone chases it "
           "down: past the school, through the market, into the duck pond.",
           "It floats! The pumpkin wins 'Best Boat' at the fair instead."),
        _s("bottle_message", "Message in a Bottle",
           "Leo finds a bottle in the river with a message: 'Is anyone there?'. He and his mum follow the "
           "river upstream to find who sent it.",
           "It is a girl across the river, the same age, and they become paper-boat pen pals."),
        _s("kite_wont_come_down", "The Kite That Wouldn't Come Down",
           "Noor's new fish-shaped kite loves the sky so much that it won't come down when it is time to go "
           "home. Noor and Dad try everything from the sand: reeling it in, whistling, waving a real fish "
           "sandwich at it.",
           "At sunset the wind goes to sleep, and the kite floats down by itself - right onto the roof of the "
           "ice-cream van, so Dad says it has good taste."),
    ],
    # Wonder seeds (parent feedback: "like someone who only needed hydrogen, living
    # on the Sun - stories they can learn something from"). The hero lives the
    # science; the plot is finding something out, with funny wrong guesses on the
    # way and the true answer told in conversation. Facts checked against NASA/ESA
    # explainers and kept round ("about", "hundreds of years").
    "space_science": [
        _w("sun_hydrogen", "The Little Flame Who Only Ate Hydrogen", "a little flame who lives deep inside the Sun",
           "The little flame eats only one thing: hydrogen, the tiniest thing there is. Every time she squeezes "
           "two bits of it together - pop! - out comes light and warmth. But she has never seen where all her "
           "light goes, so she hops onto a sunbeam to find out.",
           "The Sun is a star. Deep inside, it squeezes hydrogen together, and that makes light and heat. "
           "Sunlight takes about 8 minutes to reach Earth.",
           "After eight minutes the sunbeam lands on a sleepy child's window - and a sunflower in the garden "
           "turns its face to say thank you."),
        _w("ring_pebble", "The Snowball in Saturn's Rings", "a small chunk of ice in Saturn's rings",
           "The little ice chunk thinks Saturn's rings are a smooth, shiny road, and it wants to find the end of "
           "the road. But every 'road' it rolls onto turns out to be other ice chunks, rocks and dust - "
           "billions of them, all chatting and going round and round.",
           "Saturn's rings are not solid: they are billions of pieces of ice and rock, some as small as a grain "
           "of sand and some as big as a house, all going round Saturn.",
           "A space probe takes a photo from far away, and the little chunk sees the rings at last: it is one "
           "tiny sparkle in the most beautiful rings of all."),
        _w("red_spot_nap", "The Storm Who Wanted a Nap", "the Great Red Spot, a giant storm on Jupiter",
           "The Great Red Spot has been spinning for hundreds of years and is bigger than the whole Earth. It "
           "would love one little nap, but Jupiter has no ground to lie down on - it is all clouds and gas - "
           "and the winds keep spinning it round.",
           "Jupiter is the biggest planet. It is made mostly of gas, so there is no ground to stand on, and its "
           "Great Red Spot is a storm bigger than the whole Earth.",
           "Jupiter's moons tell it that children on Earth look at it through telescopes, so the storm decides "
           "spinning is its way of waving."),
        _w("venus_sunrise", "Sunrise on the Wrong Side", "a small explorer robot on Venus",
           "The robot lands on Venus and waits for sunrise to start exploring. It waits and waits - for months. "
           "Then the Sun finally comes up, but on the wrong side of the sky, and the robot thinks it has been "
           "put down back to front.",
           "Venus spins the other way from Earth, so the Sun rises in the west. It spins so slowly that one day "
           "on Venus is longer than its whole year.",
           "The robot has its first Venus birthday before its first Venus day is over, and has a party facing "
           "west."),
        _w("mars_blue_sunset", "The Rover Who Collected Sunsets", "a little rover on Mars",
           "The rover sends a photo of every Martian sunset home. Children on Earth write back that their "
           "sunsets are orange and pink, and the rover thinks they are joking, because everyone knows sunsets "
           "are blue.",
           "On Mars the dusty daytime sky looks pinky-orange, and the sunsets look blue - the other way round "
           "from Earth.",
           "The children send a photo of an orange sunset, the rover sends a blue one, and they pin them up "
           "side by side."),
        _w("drifting_moon", "The Moon Who Moved Away Very Slowly", "the Moon",
           "The Moon finds out it moves a tiny bit further from its best friend, the Earth, every year, and it "
           "worries. It tries to hold on: it pulls the sea towards it, it stays up all day, it asks the stars "
           "for advice.",
           "The Moon moves about 4 centimetres further from Earth every year - about as fast as your "
           "fingernails grow. Its pull is what makes the sea go in and out (the tides).",
           "It is so slow they will be together for billions more years, and every day the tide still comes "
           "in to say hello.", ages=(7, 10)),
        _w("lonely_footprint", "The Footprint Who Waited", "an astronaut's bootprint on the Moon",
           "The bootprint has stayed perfectly still on the Moon for more than fifty years, because there is "
           "no wind and no rain to wash it away. It wonders if anybody remembers it, and counts the Earthrises "
           "to pass the time.",
           "There is no wind or rain on the Moon, so footprints there can stay for millions of years.",
           "A new rover rolls up and parks right beside it, and the footprint finally has a neighbour."),
        _w("neptune_birthday", "Neptune's First Birthday", "the planet Neptune, far out in the cold",
           "Neptune is waiting for its first birthday since people found it. Mercury has a birthday every "
           "three months and keeps sending invitations. Neptune keeps going round, and round, and round.",
           "A planet's year is one trip around the Sun. Mercury's takes about 3 months; Neptune is so far away "
           "that its year takes 165 Earth years.",
           "One day, 165 years after it was found, every telescope on Earth points at Neptune at once, like "
           "birthday candles."),
        _w("rolling_uranus", "The Planet That Rolled", "the planet Uranus",
           "All the other planets spin standing up, like tops. Uranus spins lying on its side, like a ball "
           "rolling round the Sun, and it tries very hard to stand up straight for the planets' photo.",
           "Uranus spins tipped over on its side, so it looks like it is rolling around the Sun.",
           "In the photo Uranus is the only one rolling - and everyone agrees it looks like it is having the "
           "most fun."),
        _w("comet_tail", "The Comet Who Wanted to Show Her Tail", "a small comet",
           "The comet is a dirty snowball from the far, cold edge of space. As she flies near the Sun she grows "
           "a long shining tail, and she wants to show it off to the Sun. But whenever she turns to face the "
           "Sun, her tail hides behind her.",
           "A comet is a ball of ice and dust. Near the Sun it warms up and grows a tail, and the tail always "
           "points away from the Sun.",
           "The Sun laughs and says it can see the tail perfectly - it is the Sun's warmth that makes it."),
        _w("titan_raindrop", "The Slowest Raindrop", "a raindrop on Titan, Saturn's biggest moon",
           "On Titan it rains, but very slowly, so the raindrop has time to look around and chat on the way "
           "down. It is sure it is made of water, like the rain on Earth. Everyone it meets on the way down "
           "has a different idea.",
           "Titan, Saturn's biggest moon, has rain, rivers and lakes - but they are made of methane, not "
           "water, because it is so cold there. The raindrops fall slowly, like snowflakes.",
           "It lands in a big lake full of drops just like it, and finds out what it is made of.", ages=(7, 10)),
        _w("old_starlight", "The Light That Travelled a Long Time", "a little beam of light from a faraway star",
           "The little beam of light left its star long ago and has been travelling ever since, faster than "
           "anything. It wonders who will see it at the end of its journey.",
           "Light from stars takes a long time to reach us. When you look at a star, you see light that left "
           "it long ago - some of it before your grandma was born.",
           "It arrives at the very moment a child looks up and says, 'I wish I could go there.'", ages=(7, 10)),
        _w("sixteen_goodnights", "Sixteen Goodnights", "a child whose grandma is an astronaut on the space station",
           "Grandma promised to wave goodnight at sunset from the space station. But up there the sun rises "
           "and sets about sixteen times a day - so which sunset?",
           "The space station goes round the Earth about every 90 minutes, so astronauts see about 16 "
           "sunrises and 16 sunsets every day.",
           "Grandma waves at all sixteen, and the child, watching the station cross the sky, waves back at "
           "one."),
        _w("taller_dad", "Dad Grew in Space", "a child whose dad is an astronaut",
           "On a video call from space, Dad says he has grown taller, and the child is sure he is joking. They "
           "measure him against the station's door, then against a floating banana.",
           "In space, nothing squashes your back down, so astronauts grow a little taller - about 3 to 5 "
           "centimetres. They shrink back when they come home.",
           "When Dad comes home they measure again - he has shrunk back, just a banana-width taller than the "
           "child."),
        _w("space_burp", "The Burp Problem", "an astronaut on the space station",
           "The astronaut has a fizzy-drink plan to win the best-burp contest with her brother back on Earth. "
           "But in space, burps don't work properly.",
           "In space, bubbles don't rise to the top of a drink, because nothing pulls them up, so the gas "
           "doesn't come out as a burp.",
           "She wins anyway, with the longest, wobbliest hiccup ever heard on the station."),
        _w("floating_saturn", "The Planet That Would Float", "a child who loves bath-time experiments",
           "The child hears that Saturn would float, and wants to prove it. The bath experiments get bigger and "
           "sillier: a ball, a melon, Dad's beach ball.",
           "Saturn is a giant ball of gas, so light for its size that it would float in water - if there were "
           "a bath big enough.",
           "Dad draws Saturn's rings on the beach ball, and it bobs in the bath like a tiny planet."),
    ],
    "funny": [
        _s("sock_monster", "The Sock Monster's Confession",
           "Socks keep disappearing from Ollie's laundry basket. He sets a trap, and catches a very polite "
           "monster who says he can explain.",
           "The monster is building a sock castle - and invites Ollie to the opening."),
        _s("moo_day", "The Day Everything Said Moo",
           "After Grandpa's giant sneeze, every sound in the house comes out as 'moo': the doorbell, the "
           "kettle, even the dog.",
           "A second sneeze fixes everything except the cat, who decides to keep mooing."),
        _s("escaping_jelly", "Grandma's Wobbly Jelly",
           "Grandma's birthday jelly wobbles off the table and escapes through the garden. Everyone chases "
           "it with spoons.",
           "The jelly stops at Grandma's chair, as if it wanted to say happy birthday first."),
        _s("dog_cat", "The Dog Who Thought He Was a Cat",
           "Rufus the dog decides he is a cat. He tries to purr, to nap in a box, to climb the fence.",
           "The real cat tries being a dog for a day, and they both agree to swap back."),
        _s("upside_down_day", "Dad's Upside-Down Day",
           "Dad declares an Upside-Down Day: pudding first, pyjamas at lunch, goodnight at breakfast. "
           "But Ruby's new friend is coming to tea, and Ruby wants just one hour to look normal.",
           "The new friend loves it, and asks if tomorrow can be Inside-Out Day."),
        _s("rising_cake", "The Cake That Wouldn't Stop Rising",
           "Jade's cake for the school fair keeps rising: out of the tin, out of the oven, out of the "
           "kitchen window.",
           "The whole school gets a slice, and there is still enough for tomorrow."),
        _s("snowman_holiday", "The Snowman Who Wanted a Holiday",
           "Sid the snowman wants a sunny holiday like everyone else - but he knows the sun would melt him.",
           "The children take him to the supermarket freezer aisle, which he says is lovely this time of year."),
        _s("bath_ate_duck", "The Bath That Ate the Rubber Duck",
           "The plughole gurgles, and Sam's rubber duck is gone. Sam and Mum follow the gurgles round the "
           "house, pressing their ears to the walls, the sink and the washing machine, guessing where the duck "
           "is swimming now.",
           "The duck was behind the bath tap all along, and the gurgles were Dad singing in the kitchen."),
    ],
    "brave_hearts": [
        _s("first_day_dragon", "The First-Day Dragon",
           "Dina the little dragon starts school tomorrow, but she is worried she will sneeze sparks in class.",
           "She sneezes on day one - and toasts everyone's crumpets for snack time."),
        _s("hallway_names", "The Long Hallway",
           "Night-time trips to the bathroom feel long and dark to Amir. Tonight he gives every shadow on "
           "the way a friendly name.",
           "The shadow he named 'Big Bob' turns out to be Dad's coat, and he waves to it every night."),
        _s("face_in_water", "Face in the Water",
           "At swimming, Lily doesn't want to put her face in the water. She tries with a mask, then with "
           "her eyes shut, then by blowing bubbles.",
           "Blowing bubbles works, and she blows a bubble song at her teacher."),
        _s("tall_slide", "The Tallest Slide",
           "Sam climbs the tallest slide in the park, then feels wobbly and stuck at the top. A line of kids waits "
           "behind him.",
           "He counts to three with the kid behind him - and they slide down together."),
        _s("ice_cream_order", "Ordering All by Myself",
           "Mira wants to order her own ice cream for the first time, but the words keep getting stuck. "
           "She practises with her teddy all the way to the shop.",
           "She orders perfectly - and remembers to ask for a cone for teddy too."),
        _s("doctor_visit", "The Wiggly Doctor Visit",
           "Leo is nervous about the doctor. The doctor lets him listen to his own heart first, then to "
           "the doctor's, then to his teddy's.",
           "Teddy's heart says 'brave, brave, brave', according to Leo."),
        _s("new_neighbour", "Hello in Two Languages",
           "A new girl moves in next door who speaks another language. Kofi wants to say hello but "
           "doesn't know her words.",
           "They swap one word a day, starting with 'hello' and 'ball'."),
        _s("spelling_bee", "The Spelling Bee (and the Real Bee)",
           "Rosa is nervous about the class spelling bee. Then a real bee flies in during her turn.",
           "She stays calm, spells 'bee' perfectly, and the bee flies out of the window like it's clapping."),
    ],
    "friendship": [
        _s("tangled_kites", "Two Kites, One Tree",
           "Two children who have never met get their kites tangled in the same tree. They can only get "
           "them down by working together.",
           "Their kites fly better tied together, so they keep them that way."),
        _s("box_ship_captain", "Who Is Captain?",
           "Best friends Ava and Zane build a ship from a big cardboard box, but they argue about who is "
           "the captain.",
           "They make it a two-captain ship, with two steering wheels drawn on."),
        _s("giant_mouse_bench", "The Giant and the Mouse Share a Bench",
           "A giant and a mouse both want to sit on the same park bench every day at lunch.",
           "The giant sits on the bench, and the mouse sits on the giant's hat, and they share sandwiches."),
        _s("rock_kid", "The New Kid Who Collects Rocks",
           "The new boy at school only talks about rocks. Mia thinks rocks are boring - until he shows her "
           "one with a fossil snail inside.",
           "They start a rock club with two members and one snail."),
        _s("frog_duck_pals", "Pen Pals Across the Pond",
           "Frog and Duck live on opposite sides of the pond and send each other notes on lily pads. One "
           "day a note gets lost on the way.",
           "They both swim to the middle to find it, and meet in person for the first time."),
        _s("snowball_snowman", "The Snowball Rivals",
           "Two neighbours each roll the biggest snowball on the street to win the snowman contest. Their "
           "snowballs crash in the middle of the road.",
           "Stacked together, they make the tallest snowman in town, and they share first prize."),
        _s("moving_day", "Moving Day",
           "Zoe's best friend is moving to a new town. They try to think of a way to stay best friends "
           "from far away.",
           "They plant two seeds from the same sunflower, one in each garden, and send photos."),
        _s("grumpy_lighthouse", "The Lighthouse and the Foghorn",
           "The lighthouse and the foghorn don't get along: one is bright and quiet, the other loud and dark. "
           "Then a big fog rolls in.",
           "Ships need both, and from that night they work as a team."),
    ],
}

# True, kid-sized facts for space & science stories. The setting specialist
# weaves one into the story; the judge's checklist asks for "one or two real
# facts". Checked against NASA/ESA explainers; kept simple on purpose.
SPACE_FACTS = [x.learn for x in SEEDS["space_science"]] + [
    "On the Moon you would weigh about six times less than on Earth, so you could jump very high.",
    "Space is silent: there is no air to carry sound.",
    "Astronauts sleep in sleeping bags tied to the wall so they don't float away.",
    "A teaspoon of water in space floats as a wobbly ball.",
]

# Short comic scenes, each showing ONE technique, used as a second example for
# the build-up and big-moment writers. Original, in the manner of classic
# picture books (the misunderstanding logic of Winnie-the-Pooh, the rule of
# three of folk tales). Characters and words differ from every seed above.
COMIC_SCENES = [
    ("a misunderstanding the hero is sure about",
     """Goose was baking with Hen.

"We need a muffin tin," said Hen.

"Tim?" said Goose. "Who is Tim?"

"A muffin tin," said Hen.

"I will find him," said Goose, and off she went.

She asked the cow. "Are you Tim?"

"No," said the cow. "I am Daisy."

She asked the pig. "Are you Tim?"

"I am a pig," said the pig.

Goose came back very tired. "Nobody is Tim," she said.

Hen held up the tin. "This is the tin."

Goose looked at it for a long time. "Hello, Tim," she said."""),
    ("the rule of three, with a surprise on the third try",
     """Bear wanted the jar of honey on the top shelf.

First he stood on a chair. He was still too short.

Then he stood on a chair on a box. He wobbled. "Oh no," he said, and sat down fast.

Then he had his best idea. "Mouse!" he called. "Can you help me?"

Mouse ran up the shelf in two seconds.

"Here you are," said Mouse, rolling the jar to the edge.

Bear caught it. "Thank you, Mouse," he said. "Would you like some?"

"Only a little," said Mouse. "I am very small."

So Bear gave him a little. And then a little more."""),
    ("a running joke that comes back at the end",
     """Every morning, Dad sang to the toaster. "Pop, pop, toaster, pop!"

And every morning, the toast popped up.

"It only works if you sing," said Dad.

Ruby didn't believe him.

On Saturday Dad had a sore throat. He couldn't sing at all.

Ruby looked at the toaster. The toaster looked at Ruby.

"Pop, pop, toaster, pop," she sang, very quietly.

Pop! went the toast.

Dad smiled and gave a thumbs up.

That night, Ruby sang to her pillow. "Plop, plop, pillow, plop." And she plopped straight to sleep."""),
    ("a character who takes words too literally",
     """"Hop into bed," said Mum.

So Frog hopped. He hopped into bed, out of bed, and into bed again.

"Just once," said Mum.

"You said hop," said Frog.

"Now jump into your pyjamas," said Mum.

Frog looked at his pyjamas on the floor. He jumped on them.

"INTO them," said Mum, laughing.

Frog put them on, slowly, one leg at a time.

"Now," said Mum, "close your eyes."

Frog closed one eye.

"Both," said Mum.

"You didn't say both," said Frog. But he closed the other one too."""),
]


def seed_line(x: Seed) -> str:
    """How a seed is shown to the agent and the planning specialists."""
    return (x.hook + (f" Hero: {x.hero}." if x.hero else "") + f" Twist: {x.twist}"
            + (f" What the child learns: {x.learn}" if x.learn else ""))


def seeds_for(genre: str, age: int, *, avoid: Optional[set] = None, k: int = 5,
              rng: Optional[random.Random] = None) -> list[Seed]:
    """Up to k seeds for this genre and age, fresh ones first (avoid = recently used keys)."""
    pool = [s for s in SEEDS.get(genre, []) if s.ages[0] <= age <= s.ages[1]]
    if genre == "auto" or not pool:
        pool = [s for g in SEEDS.values() for s in g if s.ages[0] <= age <= s.ages[1]]
    fresh = [s for s in pool if s.key not in (avoid or set())] or pool
    rng = rng or random
    return rng.sample(fresh, min(k, len(fresh)))


def facts_for(genre: str, k: int = 4, rng: Optional[random.Random] = None) -> list[str]:
    if genre != "space_science":
        return []
    return (rng or random).sample(SPACE_FACTS, k)


def comic_scene(n: int) -> tuple[str, str]:
    return COMIC_SCENES[n % len(COMIC_SCENES)]
