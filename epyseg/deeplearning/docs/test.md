TRASH

# Training: key things to keep in mind
Since during training the output is user provided, it is particularly important to pay attention that it is of good quality (e.g. if the output is a segmentation it is important to make sure it is of good quality and unbiased as the model will try to reproduce. The model will not learn well if the segmentation is not done properly and if the segmentation is biased the model will keep the same bias.

- Does a deep learning model already exist for your segmentation ? If the answer is 'yes' then load the model otherwise you will have to build your own model.
-load an existing model: typically drag and drop a .json or .h5/.hdf5 file. Tip .JSON models are lightweight but they only contain the model architecture and not model weights. .h5 files may contain a model with or without weights or weights without a model.    
        
This a test of all markdown possibilities:

------------------------------------------

## Headings

# h1 Heading 1
## h2 Heading 2
### h3 Heading 3
#### h4 Heading 4
##### h5 Heading 5
###### h6 Heading 6

------------------------------------------

## Horizontal Rules

___

---

***

------------------------------------------

## Emphasis

**This is bold text**

__This is bold text__

*This is italic text*

_This is italic text_

~~Strikethrough~~

------------------------------------------

## Links

[link text][1]

[link with title][2]

This is [an example](http://example.com/ "Title") inline link.

[This link](http://example.net/) has no title attribute.

------------------------------------------

## Blockquotes

> Blockquotes can also be nested...
>> ...by using additional greater-than signs right next to each other...
> > > ...or with spaces between arrows.

------------------------------------------

## Indentation

  indentation 1-1

indentation 1-2
    indentation 2-1
    

------------------------------------------

## Lists

### Unordered

+ Create a list by starting a line with `+`, `-`, or `*`
+ Sub-lists are made by indenting 2 spaces:
  - Marker character change forces new list start:
    * Ac tristique libero volutpat at
    + Facilisis in pretium nisl aliquet
    - Nulla volutpat aliquam velit
+ Very easy!

### Ordered

#### Numers in sequence

1. Lorem ipsum dolor sit amet
2. Consectetur adipiscing elit
3. Integer molestie lorem at massa

#### Numers not in sequence

1. You can use sequential numbers...
1. ...or keep all the numbers as `1.`

------------------------------------------

## Images

![Minion][3]
![Stormtroopocat][4]

Like links, Images also have a footnote style syntax

![Alt text][5]

With a reference later in the document defining the URL location:



------------------------------------------

## Tables

| Option | Description |
| ------ | ----------- |
| data   | path to data files to supply the data that will be passed into templates. |
| engine | engine to be used for processing templates. Handlebars is the default. |
| ext    | extension to be used for dest files. |

Right aligned columns

| Option | Description |
| ------:| -----------:|
| data   | path to data files to supply the data that will be passed into templates. |
| engine | engine to be used for processing templates. Handlebars is the default. |
| ext    | extension to be used for dest files. |

------------------------------------------

## Code

Inline `code`

Indented code

    // Some comments
    line 1 of code
    line 2 of code
    line 3 of code


Block code "fences"

```
Sample text here...
```

Syntax highlighting

``` js
var foo = function (bar) {
  return bar++;
};

console.log(foo(5));
```

------------------------------------------


[1]: http://dev.nodeca.com
[2]: http://nodeca.github.io/pica/demo/ "title text!"
[3]: https://octodex.github.com/images/dinotocat.png
[4]: https://octodex.github.com/images/saritocat.png "The Stormtroopocat"
[5]: https://octodex.github.com/images/daftpunktocat-thomas.gif "The Dojocat"