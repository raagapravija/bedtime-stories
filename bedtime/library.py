"""The local story library: one example story per age band and tone.

The Storyteller Agent calls search_library(age, tone) and gets ONE story to copy
the *voice* of - how it is told, how simple the sentences are, how the
characters talk - never its names, place or plot. gpt-3.5 imitates an example
far more faithfully than it follows a description of a style, so a matched
example (funny for a 5-year-old, a quiet mystery for a 9-year-old) is the
strongest tone control we have.

Every story here is either an original or a simple retelling of an
out-of-copyright classic (noted in `source`). Every one passes the same
simplicity checks as the stories the app makes (tests/test_library.py).

`signature` lists the story's own names and ideas. If the agent's plan or
story uses two or more of them, the code checks say it copied the example.
"""
from __future__ import annotations

from dataclasses import dataclass

AGE_BANDS = {"5-6": (5, 6), "7-8": (7, 8), "9-10": (9, 10)}
TONES = ("funny", "cozy", "mystery", "adventure")

# Which tone fits each genre (the agent may still choose another one).
GENRE_TONE = {
    "funny": "funny", "good_manners": "funny", "healthy_habits": "funny",
    "calming": "cozy", "friendship": "cozy", "animal_friends": "cozy",
    "magical": "mystery", "space_science": "mystery",
    "fairy_tale": "adventure", "adventure": "adventure", "brave_hearts": "adventure",
}


@dataclass(frozen=True)
class LibraryStory:
    key: str
    title: str
    band: str
    tone: str
    source: str
    signature: tuple[str, ...]
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


def band_for(age: int) -> str:
    for band, (lo, hi) in AGE_BANDS.items():
        if lo <= age <= hi:
            return band
    raise ValueError(f"no age band for {age}")


LIBRARY = [
    # ------------------------------------------------------------- ages 5-6
    LibraryStory(
        key="bobo_door", title="Bobo and the Round Green Door", band="5-6", tone="funny",
        source="Retold from Winnie-the-Pooh (A. A. Milne, 1926) with new characters",
        signature=("bobo", "pip", "hoot", "honey", "round green door", "towels"),
        text="""Bobo the bear was hungry. Bobo was always a little bit hungry.

"I know," said Bobo. "I will go and see Pip."

Pip the rabbit lived in a little house under a hill. It had a round green door. It was a very small door.

Bobo knocked. Knock, knock, knock.

"Who is there?" asked Pip.

"It's me, Bobo!"

"Come in, come in," said Pip.

Bobo squeezed through the round green door. Pip had a big table. On the table there was bread. There was jam. There was a big pot of honey.

"Would you like a little snack?" asked Pip.

"Yes, please," said Bobo. "Just a little one."

Bobo ate some bread. Then he ate some jam. Then he ate the honey. All of the honey.

He licked the pot. He licked his paws. He licked his nose.

"Is there any more?" asked Bobo.

Pip looked in the pot. Pip looked in the jar. "No," said Pip. "You ate it all."

"Oh," said Bobo. "Then I'd better go home. Thank you, Pip!"

Bobo went to the round green door. He put his head out. He put his arms out. He pushed and pushed...

But his tummy did not come out.

"Pip," said Bobo. "I think I'm stuck."

Pip came to look. Bobo's head was outside. Bobo's tummy was inside. His back legs kicked in the air.

"Oh my goodness," said Pip. "You are stuck!"

"Can you pull me?" asked Bobo.

Pip held Bobo's back legs. Pip pulled. Bobo wiggled. Nothing.

Then Hoot the owl flew down from his tree. "What's this?" asked Hoot. "A bear in a door?"

"I ate too much honey," said Bobo.

"There is only one thing to do," said Hoot. "You must wait until your tummy gets smaller."

"How long will that take?" asked Bobo.

"Maybe a week," said Hoot.

"A week!" said Bobo. "But I'll get hungry!"

So Bobo waited. And waited. Every night, Hoot sat by Bobo's nose and read him a story.

"Bobo, can I hang my towels on your legs?" asked Pip.

"Yes," said Bobo. Then he laughed. The towels tickled.

Each day, Bobo's tummy got a little bit smaller. Then one morning, Pip said, "Let's try again!"

Pip grabbed Bobo's feet. Hoot grabbed Pip. Everyone pulled together.

"One, two, three... PULL!"

POP!

Bobo landed on the soft grass. Everyone fell down in a big, happy heap.

"I'm out!" said Bobo. "Thank you, everyone!"

That night, Bobo was back in his own bed. His tummy was just the right size. He took a big, slow breath in... and out. And he fell fast asleep."""),
    LibraryStory(
        key="hen_bread", title="Little Hen's Warm Bread", band="5-6", tone="cozy",
        source="Retold from the folk tale The Little Red Hen, with a kinder ending",
        signature=("little hen", "wheat", "flour", "mill", "goose", "pig"),
        text="""Little Hen found a tiny seed. It was a seed of wheat.

"Who will help me plant it?" asked Little Hen.

"Not me," said Duck. "I want to swim."

"Not me," said Goose. "I want to nap."

"Not me," said Pig. "I want to roll in the mud."

"Then I will do it myself," said Little Hen. And she did.

The sun came out. The rain came down. The wheat grew tall and gold.

"Who will help me cut the wheat?" asked Little Hen.

"Not me," said Duck.

"Not me," said Goose.

"Not me," said Pig.

So Little Hen cut it all by herself. Snip, snip, snip.

She took the wheat to the mill. The big wheel went round and round. Now she had a bag of soft white flour.

"Who will help me make bread?" asked Little Hen.

Duck looked at Goose. Goose looked at Pig. Pig looked at his muddy feet.

Little Hen mixed the flour with water. She pushed the dough and pulled the dough. Her wings got very tired.

She put the bread in the oven. Soon the whole farm smelled warm and sweet.

Duck lifted her head. "What is that smell?"

Goose opened one eye. "Is that bread?"

Pig stopped rolling. "Hot bread?"

They all ran to Little Hen's door.

"Who will help me eat this bread?" asked Little Hen.

"Me!" said Duck.

"Me!" said Goose.

"Me!" said Pig.

Little Hen put her wings on her hips. "Who helped me plant the seed?" she asked.

Nobody said a word.

"Who helped me cut the wheat?"

Nobody said a word.

Then Duck said, "We are sorry, Little Hen. We should have helped you."

"Can we help now?" asked Goose.

Little Hen thought about it. "Yes," she said. "You can help me set the table."

Duck brought the plates. Goose brought the cups. Pig washed his feet. Twice.

They all sat down together. Little Hen cut the bread into four big pieces. It was warm and soft inside.

"This is the best bread in the world," said Pig.

"Next time," said Duck, "we will help you plant the seed."

"And cut the wheat," said Goose.

Little Hen smiled.

That night, Little Hen sat in her warm little nest. Her tummy was full. Her friends were close by. She tucked her head under her wing, took one slow breath, and went to sleep."""),
    LibraryStory(
        key="moon_bite", title="Who Took a Bite of the Moon?", band="5-6", tone="mystery",
        source="Original story",
        signature=("ollie", "mama bear", "bite of the moon", "took a bite", "eating you"),
        text="""Ollie the little bear looked out of his window. The moon was big and round.

"Look, Mama," said Ollie. "The moon is round like a plate."

"It is," said Mama Bear. "Now, into bed."

The next night, Ollie looked again.

"Mama!" he said. "Someone took a bite of the moon!"

Mama Bear looked. One side of the moon was gone.

"Who took a bite?" asked Ollie. "Was it a mouse?"

"I don't think so," said Mama Bear.

"Was it a big bird?"

"No, I don't think so."

Ollie had an idea. "I will find out," he said.

The next night, Ollie took his little lamp. He took a blanket. He took a cookie, in case he got hungry.

He sat by the window and watched.

The moon was smaller again. Now it looked like a smile.

"Moon," said Ollie, "who is eating you?"

The moon did not answer. Moons never do.

Ollie watched and watched. His eyes got heavy. His head went down on the blanket.

"Time for bed, little detective," said Mama Bear, and she carried him to his room.

Night after night, the moon got smaller. Then one night, it was gone. All of it.

Ollie was very sad. "Someone ate the whole moon," he said. "Now it's dark."

Mama Bear held his paw. "Let's wait and see," she said.

The next night, Ollie looked out. And there it was. A tiny thin moon, like a fingernail.

"It's back!" said Ollie. "It's growing!"

Every night, the moon got a little bigger. A smile. Then half a plate. Then almost round.

Then one night, it was big and round again.

"Mama, how did it do that?" asked Ollie.

Mama Bear sat on his bed. "Nobody eats the moon," she said. "The moon is always there. But the sun only lights up part of it. Some nights we see a big part. Some nights we see a tiny part."

"So it's playing hide and seek?" asked Ollie.

"Just like that," said Mama Bear.

Ollie looked at the big round moon. "Goodnight, Moon," he said. "Don't hide for too long."

Mama Bear pulled up his blanket. Ollie took a slow breath in... and out. And he fell asleep, with the moon shining on his pillow."""),
    LibraryStory(
        key="mouse_lion", title="The Little Mouse and the Big Lion", band="5-6", tone="adventure",
        source="Retold from Aesop's fable The Lion and the Mouse",
        signature=("lion", "mina", "vines", "tangled", "chewed"),
        text="""Mina the mouse lived in the tall grass. Every day she ran and played.

One day she ran up a big brown hill. But it was not a hill. It was a lion!

The lion opened one eye. "Who is running on my nose?" he said.

"Oh no," said Mina. "I'm sorry, Mr Lion. I thought you were a hill."

The lion looked at her. "You are very small," he said.

"I am," said Mina. "But please let me go. One day, I will help you."

The lion laughed. It was a big, loud laugh. "A little mouse help a big lion? That's funny."

But he let her go. "Run home, little mouse," he said.

"Thank you!" said Mina. "I won't forget."

Many days went by. The lion went for a walk in the woods. He did not look where he was going.

He walked into a big bunch of vines. The vines went round his legs. The vines went round his tail.

He pulled. He wiggled. He could not get out.

"Help!" said the lion. "I'm all tangled up!"

A bird flew by. "I can't help," said the bird. "My beak is too small."

A frog hopped by. "I can't help," said the frog. "I'm too slow."

The lion sat down. "Who will help me now?" he said.

Then he heard a tiny voice. "I will!"

It was Mina the mouse.

"You?" said the lion. "But you are so small."

"Small teeth are good for chewing," said Mina.

She ran to the vines. She chewed and chewed. Chomp, chomp, chomp.

One vine broke. Then another. Then another.

The lion pulled one leg out. Then two. Then all four. Then his tail.

"I'm free!" said the lion. "Little mouse, you helped me!"

"I said I would," said Mina.

The lion bent down very low. "Thank you, my friend," he said. "I was wrong. Small friends can do big things."

"Can I ride on your back?" asked Mina.

"Yes," said the lion. "Hold on tight."

Mina sat on the lion's soft head. He walked her all the way home, through the tall grass.

That night, Mina curled up in her little bed of grass. Far away, the lion yawned a big lion yawn.

"Goodnight, little friend," he said to the stars.

And Mina took a slow breath in... and out. She fell asleep with a smile on her face."""),
    # ------------------------------------------------------------- ages 7-8
    LibraryStory(
        key="town_mouse", title="Town Mouse and Country Mouse", band="7-8", tone="funny",
        source="Retold from Aesop's fable The Town Mouse and the Country Mouse",
        signature=("town mouse", "country mouse", "barley", "cousin", "chocolate cake"),
        text="""Country Mouse lived in a little hole under an old apple tree. She had a bed of soft hay and a cupboard full of seeds.

One day her cousin came to visit. He was Town Mouse, and he wore a smart blue scarf.

"Welcome, cousin!" said Country Mouse. "Come and have dinner."

She put out her best food: barley, two peas and a big red berry.

Town Mouse nibbled a pea. "Is this all you eat?" he asked.

"It's very good barley," said Country Mouse.

"It's very plain barley," said Town Mouse. "You must come to the city with me. In the city, we eat cake!"

Country Mouse had never tasted cake. "All right," she said. "Just for one night."

They walked all day. The city was loud. Cars went beep. Bikes went ring. People's feet went stomp, stomp, stomp.

"Isn't it wonderful?" said Town Mouse.

"It's very noisy," said Country Mouse, holding her ears.

Town Mouse lived inside the wall of a big house. That night, they tiptoed into the dining room. The table was full of food. There was cheese, jelly, cookies and a huge chocolate cake.

"Help yourself," said Town Mouse.

Country Mouse took one bite of cake. It was sweet and sticky and very, very good.

Just then, the door swung open. WOOF!

A big dog ran into the room. He wagged his tail and barked at the table.

"Run!" said Town Mouse.

They ran back into the wall. Country Mouse had cake on her nose and her heart was going thump, thump, thump.

"Does this happen every night?" she asked.

"Oh yes," said Town Mouse. "You get used to it."

They waited. The dog went away. They crept back to the table. Country Mouse took a piece of cheese.

BEEP! A car honked outside. Country Mouse jumped so high that the cheese landed on her head.

Town Mouse laughed so hard he fell off the table.

In the morning, Country Mouse packed her things.

"Leaving already?" asked Town Mouse.

"Thank you for the cake," said Country Mouse. "It was the best thing I ever ate. But I like my quiet home."

"Even with plain barley?"

"Even with plain barley," said Country Mouse. "Come and visit me. I'll make you a berry pie."

She walked all the way home. When she saw her apple tree, she ran the last bit.

That night she lay in her bed of soft hay. There were no cars, no dogs, no stomping feet. Only crickets, singing slowly. She took a long breath in and out, and fell asleep."""),
    LibraryStory(
        key="shoemaker_elves", title="The Shoemaker's Little Helpers", band="7-8", tone="cozy",
        source="Retold from the Brothers Grimm, The Elves and the Shoemaker",
        signature=("shoemaker", "elves", "elf", "leather", "workbench", "hans", "greta"),
        text="""Once there was a kind old shoemaker named Hans. He made good, strong shoes. But business was slow, and one night he had just enough leather for one last pair.

He cut out the leather and laid the pieces on his workbench.

"I'll sew them in the morning," he told his wife, Greta.

"Don't worry," said Greta. "Something good will happen. I can feel it."

Then they went to bed.

In the morning, Hans came downstairs. He stopped and rubbed his eyes.

On the workbench stood a pair of shoes. They were finished. Every stitch was tiny and perfect.

"Greta!" he called. "Come and look!"

"Did you get up in the night?" asked Greta.

"No," said Hans. "I slept like a log."

That day a lady came in and saw the shoes. "They're beautiful," she said. She paid so much that Hans could buy leather for two more pairs.

That night he cut out the leather for two pairs and left it on the bench. In the morning, there were two pairs of shoes, finished and shining.

It happened again and again. Every night Hans left leather. Every morning there were shoes.

"Look at these stitches," said Hans. "They're so small. Who could sew like this?"

"Someone with very small hands," said Greta.

Soon the shop was full of happy people.

"Who is helping us?" asked Greta one evening. "We must find out."

So that night they hid behind the big coat in the corner of the shop. The clock struck twelve.

Two tiny elves climbed up onto the bench. They had no shoes and no coats. Their clothes were thin and full of holes.

"Brr, it's cold tonight," said the first elf.

"Sew fast and we'll be warm," said the second elf.

They picked up the leather and began to sew. Their little hands moved so fast. Tap, tap, tap went their tiny hammers.

When the shoes were done, the elves jumped down and ran out under the door.

"They have worked so hard for us," said Greta. "And they looked so cold."

"Let's make them something," said Hans. "Shirts and coats."

"And boots," said Greta. "Nobody should have cold feet."

The next day, Greta sewed two tiny shirts and two tiny coats. Hans made two tiny pairs of boots, the smallest boots he had ever made.

That night they put the little clothes on the bench instead of leather. Then they hid again.

At twelve o'clock the elves climbed up. They saw the clothes and stopped.

"For us?" said one.

"For us!" said the other.

They put on the shirts and coats and boots. They danced around the bench and laughed. Then they hopped down and ran out into the night.

The elves never came back. But the shop stayed busy, and Hans and Greta were never poor again.

Every night before bed, Greta left a little bowl of milk by the door, just in case.

"Goodnight, little helpers," she said. Then she and Hans went up to bed, warm and happy, and slept until morning."""),
    LibraryStory(
        key="missing_socks", title="The Mystery of the Missing Socks", band="7-8", tone="mystery",
        source="Original story",
        signature=("sock", "socks", "biscuit", "maya", "detective", "case closed"),
        text="""It started on Monday. Maya's red sock was gone.

"Mum, have you seen my red sock?" she asked.

"It must be in the wash," said Mum.

On Tuesday, her striped sock was gone too. On Wednesday, both her blue socks disappeared.

"Socks don't just walk away," said Maya. "This is a mystery."

She got her notebook and a pencil. On the first page she wrote: THE CASE OF THE MISSING SOCKS.

"Are you a detective now?" asked Dad.

"Yes," said Maya. "And I'm going to find the sock thief."

First, she looked for clues. Under her bed, there was dust, a lost crayon and one old book. No socks.

In the laundry basket, there were towels and Dad's shirts. No socks.

"Maybe the washing machine eats them," said her little brother Sam.

Maya wrote that down. Then she crossed it out. "Washing machines don't have mouths," she said.

On Thursday, Maya set a trap. She put one yellow sock on the floor by her door. Then she hid behind the curtain and waited.

Nothing happened. She waited some more. Her legs got tired.

Then she heard a soft sound. Pad, pad, pad.

Something small and brown came down the hall. It had floppy ears and a waggy tail.

It was Biscuit, their new puppy.

Biscuit sniffed the yellow sock. He picked it up gently in his mouth. Then he trotted away down the hall.

Maya followed on tiptoe. Biscuit went into the kitchen. He went past his water bowl. He went behind the big armchair in the corner.

Maya peeked behind the chair.

There, in a cozy pile, were all her socks. The red one. The striped one. The blue ones. Even two of Dad's.

And in the middle of the pile lay Biscuit, with the yellow sock under his chin.

"Biscuit!" said Maya. "You're the sock thief! Mum, Dad, come and see!"

Biscuit thumped his tail.

Mum came in and looked. "He's made himself a bed," she said. "Socks smell like us. I think he wanted to sleep close to his family."

Maya knelt down and stroked his soft ears. "You don't need to steal socks," she said. "You can sleep near us."

That night, Dad put Biscuit's basket at the end of Maya's bed. Maya gave him one old sock to keep.

In her notebook she wrote: CASE CLOSED.

Biscuit curled up with his sock. Maya pulled her blanket up to her chin. They both took a long, slow breath, and fell asleep at the same time."""),
    LibraryStory(
        key="tortoise_hare", title="The Tortoise and the Hare", band="7-8", tone="adventure",
        source="Retold from Aesop's fable The Tortoise and the Hare",
        signature=("tortoise", "hare", "race", "finish line", "slow and steady"),
        text="""Hare was the fastest animal in the whole meadow, and he liked to tell everyone.

"Nobody can beat me," he said. "I'm as fast as the wind."

Tortoise looked up from his lettuce. "I'll race you," he said.

All the animals laughed. Even Tortoise's friend Duck laughed a little.

"You?" said Hare. "You're the slowest animal in the meadow!"

"Maybe," said Tortoise. "But I'd still like to race."

So they made a plan. The race would go from the big oak tree, over the hill, past the pond, to the old red gate.

The next morning, all the animals came to watch. Owl held up a leaf. "Ready... set... go!"

Hare shot off like a rocket. In a blink he was over the hill.

Tortoise took one step. Then another step. Then another.

"Come on, Tortoise!" called Duck.

Hare ran past the pond. He looked back. Tortoise was a tiny dot on top of the hill.

"This is too easy," said Hare. "I have lots of time. I'll have a little rest."

He lay down under a shady tree. The grass was soft. The sun was warm. A bee hummed nearby.

Soon Hare was fast asleep.

Tortoise kept walking. Step, step, step. Down the hill. Step, step, step. Past the pond. His legs were tired, but he did not stop.

He walked right past the shady tree. Hare was snoring.

"Shh," said Tortoise to a frog. "Don't wake him."

On and on he went. Now he could see the old red gate. The animals were cheering.

"Go, Tortoise, go!"

The noise woke Hare up. He jumped to his feet. "What? Where is Tortoise?"

He saw Tortoise near the gate. Hare ran faster than he had ever run.

But it was too late. Tortoise stepped across the finish line, one slow step ahead.

The meadow went wild. Duck flapped. Owl hooted. The rabbits danced.

Hare came puffing up. "How did you win?" he asked. "You're so slow!"

"I didn't stop," said Tortoise. "Slow and steady can still get there."

Hare looked at his feet. "I'm sorry I laughed at you," he said. "Will you race me again next week?"

"Yes," said Tortoise. "But no naps this time."

Everyone laughed, even Hare.

That evening, the sky turned pink and gold. Tortoise walked slowly home and crawled into his warm spot under the bushes. It had been a very long day.

He pulled his head into his shell, took a slow, sleepy breath, and was asleep before the first star came out."""),
    # ------------------------------------------------------------ ages 9-10
    LibraryStory(
        key="secret_garden", title="The Secret Garden", band="9-10", tone="mystery",
        source="Simplified retelling of a scene from The Secret Garden (F. H. Burnett, 1911)",
        signature=("mary", "robin", "ivy", "secret garden", "moor"),
        text="""Mary was a nine-year-old girl. She had just come to live at a giant, quiet house with one hundred rooms, next to a wide, windy moor. Mary was often lonely. Her favorite thing to do was to explore the big walled gardens around the old house.

She had heard a story from the servants about a very special garden. It was a secret garden. Someone had locked it up for ten whole years, and buried its key somewhere deep in the earth.

One chilly afternoon, Mary was walking along one of the high stone walls. The ivy grew so thick on the walls that it looked like a heavy green curtain. As she walked, a tiny bird with a bright red chest hopped along the path beside her. It was a friendly robin.

"Hello, little robin," Mary said.

The robin looked at Mary with his bright, clever eyes. He chirped a sweet, cheerful little tune, as if he were trying to tell her a secret.

"Do you know where the door is?" Mary whispered.

The robin hopped onto the stone wall and flew up to a thick branch of ivy. The wind blew across the moor and lifted the heavy ivy leaves away from the stone. Mary gasped. Under the green leaves was an old iron door handle.

Mary's heart beat fast. She reached her hand through the leaves and turned the handle. With a slow, heavy creak, the old door swung open.

Mary stepped through the doorway and closed the door softly behind her. She was standing inside the secret garden.

It was the most mysterious and quiet place she had ever seen. High stone walls went all the way around it and kept out the cold, howling wind. Soft, faded winter grass covered the ground. Old rose bushes climbed up the walls like thin lace. Their branches were gray and still. Everything looked like it was under a sleeping spell.

It was very, very quiet inside the garden. It felt like a secret room under the sky. Mary walked on tiptoe across the grass. She did not want to wake the sleeping plants.

She looked closely at the dark earth near the roots of an old tree. Tiny, sharp green points were peeking through the brown soil. They were little spring flowers, waiting for the spring to come.

"You're not dead at all," Mary whispered. "You're just resting."

The little robin flew down to a low rose branch near Mary. He tucked his head in, and his red feathers puffed out until he looked like a soft round ball. He let out one last, very quiet chirp. It sounded like a sleepy goodnight.

Mary sat down on a smooth stone bench under a big willow tree. The air inside the garden felt calm, safe and warm. The busy thoughts in her head began to slow down, just like the resting earth.

"Goodnight, little garden," Mary whispered.

Mary walked back through the secret door and slipped into her warm bedroom in the big house. She climbed into her bed and pulled the heavy blankets up to her chin. She closed her eyes and thought about the quiet, hidden garden. Then Mary drifted off into a long, sleepy rest."""),
    LibraryStory(
        key="princess_pea", title="The Princess and the Pea", band="9-10", tone="funny",
        source="Retold from Hans Christian Andersen, The Princess and the Pea (1835)",
        signature=("princess", "pea", "mattresses", "quilts", "prince leo"),
        text="""Prince Leo wanted to marry a real princess. The trouble was, he couldn't tell which princesses were real.

He travelled all over the world to meet them. One princess laughed too loudly. One princess only talked about her ponies. One princess kept her crown on in the bath. Leo came home tired and grumpy.

"There's no such thing as a real princess," he told his mother, the Queen.

"Nonsense," said the Queen. "You just haven't found one yet."

That night, a terrible storm hit the castle. Thunder boomed and rain poured down the windows.

Then came a knock at the big front door.

The old King opened it himself. Outside stood a girl, soaked from head to toe. Water ran out of her hair and into her shoes and out of her heels.

"Good evening," she said politely. "I'm a princess. My carriage broke a wheel. Could I stay the night?"

"A princess?" said the Queen. She looked at the girl's muddy dress. "We'll see about that."

While the girl dried off by the fire, the Queen marched upstairs with a plan.

She took all the bedding off the guest bed. On the bare wooden boards she placed one small, green, dried pea.

Then she called the servants. "Mattresses!" she said. "Bring me every mattress in the castle."

They piled up twenty mattresses on top of the pea. Then the Queen asked for twenty feather quilts and piled them on top of the mattresses. The bed was so tall that the princess needed a ladder to get into it.

"Sleep well, my dear," said the Queen, with a small smile.

In the morning, everyone gathered for breakfast. The princess came down the stairs slowly, rubbing her back.

"How did you sleep?" asked the Queen.

"Oh, terribly!" said the princess. "I hardly closed my eyes all night. There was something hard in the bed. It felt like a rock. I'm black and blue all over!"

The Queen clapped her hands. The King dropped his spoon. Prince Leo nearly fell off his chair.

"Only a real princess," said the Queen, "could feel one tiny pea through twenty mattresses and twenty quilts!"

Prince Leo laughed for the first time in weeks. "Would you like to stay a little longer?" he asked the princess. "I promise to take the pea out of your bed."

"I'd like that," said the princess.

They became the best of friends, and later they were married. The pea was put in a glass box in the castle museum, where you can still see it today, unless someone has eaten it.

That night, the princess slept on just one mattress, soft and flat and pea-free. The storm was gone, the stars were out, and she fell asleep before she had even finished yawning."""),
    LibraryStory(
        key="heidi_hayloft", title="Heidi's Bed in the Hay", band="9-10", tone="cozy",
        source="Simplified retelling of a scene from Heidi (Johanna Spyri, 1881)",
        signature=("heidi", "hayloft", "goats", "round window", "grandfather"),
        text="""Heidi was five when her aunt took her up the mountain to live with her grandfather. The path was long and steep, and by the time they reached the little wooden hut, the sun was sinking behind the peaks.

Her grandfather was a tall man with a big grey beard. He did not smile much. He looked at Heidi for a long time.

"So you've come to stay," he said.

"Yes," said Heidi. "Can I see inside your house?"

He opened the door. Inside there was one room with a table, a chair, a big bed and a fireplace with a black pot hanging over it. A ladder leaned against the wall.

"What's up there?" Heidi asked.

"The hayloft," said her grandfather.

Heidi was up the ladder before he could say another word. The loft was full of fresh, sweet-smelling hay. At the end there was a little round window, and through it she could see the whole valley, far below.

"I'll sleep here!" she called down. "It's the best bedroom in the world."

Her grandfather climbed halfway up and looked. Something in his face went soft.

"Then you'll need a sheet," he said.

They made the bed together. Heidi piled the hay into a soft mound and patted it flat. Her grandfather brought up a thick, rough sheet and spread it over the top. Heidi made a hay pillow at one end.

"Now it needs a blanket," she said.

Her grandfather went down and came back with a heavy sack. "Will this do?"

Heidi laughed. "It's perfect."

Then they had supper. Her grandfather toasted a big piece of cheese over the fire until it was golden and soft. He spread it on thick bread and poured Heidi a bowl of warm goat's milk.

"Is this milk from your goats?" asked Heidi.

"From Little Swan and Little Bear," he said. "You'll meet them tomorrow."

Heidi drank every drop. It was the best supper she had ever eaten.

When the sky was dark, the wind began to blow. It whistled around the hut and made the old fir trees creak and sway.

Heidi climbed up into her hay bed. Through the round window she could see a sky full of stars, brighter than any she had seen in the town.

"Goodnight, Grandfather," she called.

There was a long pause. "Goodnight, Heidi," he said.

Later, the old man climbed the ladder with his lamp, quietly, to check on her. Heidi lay fast asleep in the hay, her cheeks pink, one arm under her head. The wind sang outside, the stars shone through the round window, and she was dreaming about goats."""),
    LibraryStory(
        key="mole_river", title="Mole's First Day on the River", band="9-10", tone="adventure",
        source="Simplified retelling of a scene from The Wind in the Willows (Kenneth Grahame, 1908)",
        signature=("mole", "rat", "picnic basket", "oars", "spring cleaning", "rowing"),
        text="""It was spring, and Mole had been cleaning his little underground house all morning. He swept and he dusted and he scrubbed until his arms ached and his nose was full of dust.

Suddenly he threw down his brush. "Bother spring cleaning!" he said. "I want to go outside."

He scrambled up his tunnel and popped out into the warm sunshine. The air smelled of new grass. Birds were building nests. Everything was busy and bright.

Mole wandered across the meadow until he came to something he had never seen before: a river. It was wide and shining, and it chattered and gurgled as it hurried along.

On the far bank, a face looked out of a round hole. It had whiskers, bright eyes and small neat ears. It was Water Rat.

"Hello, Mole!" called Rat. "Would you like to come over?"

"How?" asked Mole.

Rat untied a little blue boat and rowed across. Mole climbed in carefully.

"I've never been in a boat before," Mole admitted.

Rat stopped rowing and stared at him. "Never been in a boat? Then you've never really lived! There is nothing, absolutely nothing, as good as messing about in boats."

They floated down the river together. Rat pointed out the willow trees, the kingfisher's nest and the otter's favourite fishing spot. Then they stopped on a grassy bank and Rat opened a fat picnic basket.

"What's inside?" asked Mole.

"Cold chicken, ham, salad, bread rolls, pickles and lemonade," said Rat, all in one breath.

"Oh, stop," said Mole happily. "That's far too much."

After lunch, Mole felt brave. "Rat," he said, "please let me row now."

"Not yet," said Rat. "Wait until you've had a few lessons."

But Mole could not wait. He jumped up and grabbed the oars. He dug them into the water too hard, the boat lurched, and in a second Mole was sitting in the river with only his head sticking out.

Rat laughed, then reached down and pulled him out, dripping. "Rowing lesson number one," he said. "Sit down first."

Mole was so embarrassed he couldn't speak. But Rat just wrapped him in a picnic blanket and rowed them both back to his snug house by the river.

That evening, Mole sat by Rat's fire in a borrowed dressing gown, warm and dry at last. Rat told him river stories until the room grew dim.

"Will you teach me to row?" Mole asked sleepily. "And swim?"

"Every day," said Rat.

Mole climbed into the spare bed. Outside, the river was still talking softly in the dark. He listened to it for a moment, smiled, and fell asleep."""),
]

BY_KEY = {s.key: s for s in LIBRARY}


def search_library(age: int, tone: str) -> LibraryStory:
    """One story for this age band and tone (the nearest band if needed)."""
    band = band_for(age)
    tone = tone if tone in TONES else "cozy"
    for story in LIBRARY:
        if story.band == band and story.tone == tone:
            return story
    return next(s for s in LIBRARY if s.band == band)
