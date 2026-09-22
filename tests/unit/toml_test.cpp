#include "linnet/package/toml.hpp"

#include "test.hpp"

using namespace linnet;

TEST("toml: manifest-style document") {
    const auto document = toml::parse("# comment\n"
                                      "[package]\n"
                                      "name = \"example-model\"\n"
                                      "version = \"0.1.0\" # trailing\n"
                                      "count = 1_000\n"
                                      "negative = -7\n"
                                      "flag = true\n"
                                      "\n"
                                      "[dependencies]\n"
                                      "foo = { path = \"../foo\" }\n"
                                      "bar = { git = \"https://x\", rev = \"abc\" }\n"
                                      "\n"
                                      "[[case]]\n"
                                      "file = 'valid/001.linnet'\n"
                                      "tags = [\"a\",\n  \"b\", # note\n]\n"
                                      "\n"
                                      "[[case]]\n"
                                      "file = \"invalid/101.linnet\"\n"
                                      "code = \"E2201\"\n"
                                      "nested.key = \"x\\n\"\n");
    CHECK(document.has_value());
    const toml::Table& root = *document;
    const toml::Value package{root, 0};
    CHECK_EQ(*package.find("package")->find("name")->as_string(), "example-model");
    CHECK_EQ(*package.find("package")->find("count")->as_integer(), 1000);
    CHECK_EQ(*package.find("package")->find("negative")->as_integer(), -7);
    CHECK(*package.find("package")->find("flag")->as_bool());
    CHECK_EQ(*package.find("dependencies")->find("foo")->find("path")->as_string(), "../foo");
    const toml::Array& cases = *package.find("case")->as_array();
    CHECK_EQ(cases.size(), 2U);
    CHECK_EQ(*cases[0].find("file")->as_string(), "valid/001.linnet");
    CHECK_EQ(cases[0].find("tags")->as_array()->size(), 2U);
    CHECK_EQ(*cases[1].find("code")->as_string(), "E2201");
    CHECK_EQ(*cases[1].find("nested")->find("key")->as_string(), "x\n");
    CHECK_EQ(cases[1].line, 19U);
    CHECK(package.find("missing") == nullptr);
    CHECK(package.find("package")->find("name")->find("x") == nullptr);
}

TEST("toml: errors carry a line and never crash") {
    const char* cases[] = {
        "key\n",
        "key = \n",
        "key = \"open\n",
        "key = 1.5\n",
        "key = 1979-05-27\n",
        "key = \"\"\"multi\"\"\"\n",
        "[a\n",
        "a = 1\na = 2\n",
        "[t]\n[t]\n",
        "a = 1\n[a]\n",
        "a = [1, 2\n",
        "a = { b = 1 c = 2 }\n",
        "a = \"\\q\"\n",
        "a = 99999999999999999999\n",
        "[",
        "\"",
    };
    for (const char* text : cases) {
        const auto document = toml::parse(text);
        CHECK(!document.has_value());
        if (!document) {
            CHECK(document.error().line >= 1);
            CHECK(!document.error().message.empty());
        }
    }
    CHECK_EQ(toml::parse("[t]\n[t]\n").error().line, 2U);
}

TEST("toml: empty and comment-only documents") {
    CHECK(toml::parse("")->empty());
    CHECK(toml::parse("# nothing\n\n")->empty());
    CHECK(toml::parse("a = {}\n")->at("a").as_table()->empty());
    CHECK(toml::parse("a = []\n")->at("a").as_array()->empty());
}
