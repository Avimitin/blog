+++
title = 'Debugging in function chain calling'
date = '2023-02-28'
tag = ['Rust', 'trait']
author = 'sh1marin'
+++

Sometime I will write some code like:

```rust
let client = reqwest::Client::new();
let res = client
    .post("example...")
    .json(&payload)
    .send()
    .await
    .unwrap()
    .json::<Response>()
    .await
    .unwrap();
```

And there might be some error during the query, then the final json parser will get an
unexpected JSON response that I need to figure out the actual content. But it is
inconvenient to have to rewrite the code and break the chain call to debug.

```rust
let response = client.post(...).response();
println!("{response:?}")
```

What if we can...

```rust
// ...
    .send()
    .debug()
    .json()
    .debug()
```

## Solution

We can inherit and extend the `std::fmt::Debug` trait:

```rust
pub trait DebugExt {
    fn debug(&self) -> &Self
    where
        Self: Debug,
    {
        println!("{self:?}");
        self
    }
}

impl<D: Debug> DebugExt for D {}
```

Then any struct that has the `Debug` trait implied can automatically have
`DebugExt` trait implied, so we can put `.debug()` into the chain
and inspect the value without breaking the existing code.

Playground: <https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=d283e02ea1b1041e04b21fc478f10271>

## Credit

Thanks to [@SpriteOvO](https://github.com/SpriteOvO) for teaching me about the trait inheritance part.
